# Author: Arnaud Ferré
# INRAE, Université Paris-Saclay
# arnaud.ferre@inrae.fr
# Description: Train a neural linear classifier to normalize mentions (on BB4 normalization task).
# Embed mentions and tags (average of vectors of tokens in expression).
# Then, learn to project vectors of mentions on vectors of associated tags.


#######################################################################################################
# Imports:
#######################################################################################################

# External dependencies:
import os
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tensorflow.keras import layers, models, Model, Input, regularizers, optimizers, metrics, losses, initializers, \
    backend, callbacks, activations
from gensim.models import KeyedVectors, Word2Vec
from scipy.spatial.distance import cosine, euclidean, cdist
from pronto import Ontology

# Internal libraries:
import sys
import numpy
import copy
import json
from os import listdir
from os.path import isfile, join, splitext


#######################################################################################################
# Functions:
#######################################################################################################

###################################################
# Ontological tools:
###################################################

def loader_ontobiotope(filePath):
    """
    Description: A loader of OBO ontology based on Pronto lib.
    (maybe useless...)
    :param filePath: Path to the OBO file.
    :return: an annotation ontology in a dict (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    """
    dd_obt = dict()
    onto = Ontology(filePath)
    for o_concept in onto:
        dd_obt[o_concept.id] = dict()
        dd_obt[o_concept.id]["label"] = o_concept.name
        dd_obt[o_concept.id]["tags"] = list()

        for o_tag in o_concept.synonyms:
            dd_obt[o_concept.id]["tags"].append(o_tag.desc)

        dd_obt[o_concept.id]["parents"] = list()
        for o_parent in o_concept.parents:
            dd_obt[o_concept.id]["parents"].append(o_parent.id)

    return dd_obt


def is_desc(dd_ref, cui, cuiParent):
    """
    Description: A function to get if a concept is a descendant of another concept.
    Here, only used to select a clean subpart of an existing ontology (see select_subpart_hierarchy method).
    """
    result = False
    if "parents" in dd_ref[cui].keys():
        if len(dd_ref[cui]["parents"]) > 0:
            if cuiParent in dd_ref[cui]["parents"]:  # Not working if infinite is_a loop (normally never the case!)
                result = True
            else:
                for parentCui in dd_ref[cui]["parents"]:
                    result = is_desc(dd_ref, parentCui, cuiParent)
                    if result:
                        break
    return result


def select_subpart_hierarchy(dd_ref, newRootCui):
    """
    Description: By picking a single concept in an ontology, create a new sub ontology with this concept as root.
    Here, only used to select the habitat subpart of the Ontobiotope ontology.
    """
    dd_subpart = dict()
    dd_subpart[newRootCui] = copy.deepcopy(dd_ref[newRootCui])
    dd_subpart[newRootCui]["parents"] = []

    for cui in dd_ref.keys():
        if is_desc(dd_ref, cui, newRootCui):
            dd_subpart[cui] = copy.deepcopy(dd_ref[cui])

    # Clear concept-parents which are not in the descendants of the new root:
    for cui in dd_subpart.keys():
        dd_subpart[cui]["parents"] = list()
        for parentCui in dd_ref[cui]["parents"]:
            if is_desc(dd_ref, parentCui, newRootCui) or parentCui == newRootCui:
                dd_subpart[cui]["parents"].append(parentCui)

    return dd_subpart


###################################################
# BB4 normalization dataset loader:
###################################################

def loader_one_bb4_fold(l_repPath):
    """
    Description: Load BB4 data from files.
    WARNING: OK only if A1 file is read before its A2 file (normally the case).
    :param l_repPath: A list of directory path containing set of A1 (and possibly A2) files.
    :return:
    """
    ddd_data = dict()

    i = 0
    for repPath in l_repPath:

        for fileName in listdir(repPath):
            filePath = join(repPath, fileName)

            if isfile(filePath):

                fileNameWithoutExt, ext = splitext(fileName)

                if ext == ".a1":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt not in ddd_data.keys():

                            ddd_data[fileNameWithoutExt] = dict()
                            for line in file:

                                l_line = line.split('\t')

                                if l_line[1].split(' ')[0] == "Title" or l_line[1].split(' ')[0] == "Paragraph":
                                    pass
                                else:
                                    exampleId = "bb4_" + "{number:06}".format(number=i)

                                    ddd_data[fileNameWithoutExt][exampleId] = dict()

                                    ddd_data[fileNameWithoutExt][exampleId]["T"] = l_line[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["type"] = l_line[1].split(' ')[0]
                                    ddd_data[fileNameWithoutExt][exampleId]["mention"] = l_line[2].rstrip()

                                    if "cui" not in ddd_data[fileNameWithoutExt][exampleId].keys():
                                        ddd_data[fileNameWithoutExt][exampleId]["cui"] = list()

                                    i += 1

                elif ext == ".a2":

                    with open(filePath, encoding="utf8") as file:

                        if fileNameWithoutExt in ddd_data.keys():

                            for line in file:
                                l_line = line.split('\t')

                                l_info = l_line[1].split(' ')
                                Tvalue = l_info[1].split(':')[1]

                                for id in ddd_data[fileNameWithoutExt].keys():
                                    if ddd_data[fileNameWithoutExt][id]["T"] == Tvalue:
                                        if ddd_data[fileNameWithoutExt][id]["type"] == "Habitat" or \
                                                ddd_data[fileNameWithoutExt][id]["type"] == "Phenotype":
                                            cui = "OBT:" + l_info[2].split(':')[2].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"].append(cui)
                                        elif ddd_data[fileNameWithoutExt][id]["type"] == "Microorganism":
                                            cui = l_info[2].split(':')[1].rstrip()
                                            ddd_data[fileNameWithoutExt][id]["cui"] = [cui]  # No multi-normalization for microorganisms

    return ddd_data


def extract_data(ddd_data, l_type=[]):
    """

    :param ddd_data:
    :param l_type:
    :return:
    """
    dd_data = dict()

    for fileName in ddd_data.keys():
        for id in ddd_data[fileName].keys():
            if ddd_data[fileName][id]["type"] in l_type:
                dd_data[id] = copy.deepcopy(ddd_data[fileName][id])

    return dd_data


###################################################
# An accuracy function:
###################################################

def accuracy(dd_pred, dd_resp):
    totalScore = 0.0

    for id in dd_resp.keys():
        score = 0.0
        l_cuiPred = dd_pred[id]["pred_cui"]
        l_cuiResp = dd_resp[id]["cui"]
        if len(l_cuiPred) > 0:  # If there is at least one prediction
            for cuiPred in l_cuiPred:
                if cuiPred in l_cuiResp:
                    score += 1
            score = score / max(len(l_cuiResp), len(l_cuiPred))  # multi-norm and too many pred

        totalScore += score  # Must be incremented even if no prediction

    totalScore = totalScore / len(dd_resp.keys())

    return totalScore


###################################################
# Preprocessing tools:
###################################################

def lowercaser_mentions(dd_mentions):
    dd_lowercasedMentions = copy.deepcopy(dd_mentions)
    for id in dd_lowercasedMentions.keys():
        dd_lowercasedMentions[id]["mention"] = dd_mentions[id]["mention"].lower()
    return dd_lowercasedMentions


def lowercaser_ref(dd_ref):
    dd_lowercasedRef = copy.deepcopy(dd_ref)
    for cui in dd_ref.keys():
        dd_lowercasedRef[cui]["label"] = dd_ref[cui]["label"].lower()
        if "tags" in dd_ref[cui].keys():
            l_lowercasedTags = list()
            for tag in dd_ref[cui]["tags"]:
                l_lowercasedTags.append(tag.lower())
            dd_lowercasedRef[cui]["tags"] = l_lowercasedTags
    return dd_lowercasedRef


#######################################################################################################
# Functions:
#######################################################################################################

def dense_layer_method_training(dd_train, dd_ref, embeddings):
    """
    Description: Train a neural linear classifier. Embed mentions and tags (average of vectors of tokens in expression).
    Then, learn to project vectors of mentions on vectors of associated tags.
    :param dd_train: training data (format: mention_id (string): {'T': string ('T?'),
    'type': type of entity (ex: 'Habitat'), 'mention': surface form (can be processed), 'cui': concept ID list})
    :param dd_ref: annotation ontology (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    :param embeddings: Gensim model for Word2Vec embeddings.
    :return: a trained TensorFlow model.
    (to try: erase null mention, i.e. mention with only out-of-vocabulary tokens)
    """

    ######
    # Initialization (some are specific to Gensim/Word2Vec)
    ######
    nbMentions = len(dd_train.keys())
    vocabSize = len(embeddings.key_to_index)
    sizeVST = embeddings.vector_size
    vocab = embeddings.key_to_index
    sizeVSO = len(dd_ref.keys())
    print("vocab size:", vocabSize, "- embeddings size:", sizeVST, "- number of concepts:", sizeVSO)

    ######
    # Built mention embeddings from train set:
    ######
    d_mentionVectors = dict()
    for id in dd_train.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_train[id]["mention"].split()  # A simple split on space...
        for token in l_tokens:
            if token in vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
                # (each token vector is unit-normalized, which commonly improves results)
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)  # mention vector = average of their token vectors

    ######
    # Build labels/tags embeddings from ontology:
    ######
    # Initialization:
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)  # surface forms of label are used as keys.
        nbLabtags += 1
        if "tags" in dd_ref[cui].keys():  # In my data structure, there is a preferred label and possibly some tags.
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)  # idem, surface form of tags are used.

    # Calculation (same as mention embeddings):
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in vocab:
                        dd_conceptVectors[cui][tag] += (embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))

    ######
    # Build training matrix:
    ######
    X_train = numpy.zeros((nbMentions, sizeVST))
    Y_train = numpy.zeros((nbMentions, sizeVST))

    for i, id in enumerate(dd_train.keys()):
        toPredCui = dd_train[id]["cui"][0]  # For each mention, take only one concept to train (even if many)
        X_train[i] = d_mentionVectors[id]

        if toPredCui in dd_conceptVectors.keys():
            for j, tag in enumerate(dd_conceptVectors[toPredCui].keys()):
                if dd_conceptVectors[toPredCui][tag].any():
                    Y_train[i] = dd_conceptVectors[toPredCui][tag]
                    break  # Just taking the first label/tag which has a non-null vector (if no one, will be null)

    del d_mentionVectors

    ######
    # Build neural architecture:
    ######
    inputLayer = Input(shape=(sizeVST))
    # A simple dense layer, but parameters are initialized with Identity (is very effective here: +13pts):
    denseLayer = (layers.Dense(sizeVST, activation=None, kernel_initializer=initializers.Identity()))(inputLayer)
    TFmodel = Model(inputs=inputLayer, outputs=denseLayer)

    TFmodel.summary()
    TFmodel.compile(optimizer=optimizers.Nadam(), loss=losses.CosineSimilarity(), metrics=['cosine_similarity', 'logcosh'])
    # (Cosine loss works well here)

    ######
    # Training ("early stopping" stops training if loss does not decrease of "min_delta" during "patience" epochs):
    ######
    callback = callbacks.EarlyStopping(monitor='logcosh', patience=5, min_delta=0.0001)
    history = TFmodel.fit(X_train, Y_train, epochs=200, batch_size=64, callbacks=[callback], verbose=1)
    # If you want to analyse the loss for the training:
    # plt.plot(history.history['logcosh'], label='logcosh')
    # plt.show()

    return TFmodel

######

def dense_layer_method_predicting(model, dd_mentions, dd_ref, embeddings):
    """
    Description: Take in input a TensorFlow model, project vectors of mentions with it.
    Then, calculate the distances between a projected vector of mention and each vector of tag of concepts.
    Finally, for each mention, predict the concept which has the nearest tag.
    :param model: a trained TensorFlow model.
    :param dd_mentions: mention dict (format: mention_id (string): {'T': string ('T?'),
    'type': type of entity (ex: 'Habitat'), 'mention': surface form (can be processed), 'cui': concept ID list})
    :param dd_ref: dict for annotation ontology (format: concept ID (string): {'label': preferred tag,
    'tags': list of other tags, 'parents': list of parents concepts}
    :param embeddings: Gensim model for Word2Vec embeddings.
    :return: dict of results (format: mention ID: {'pred_cui': list of predicted concept ID}).
    """

    ######
    # Initialization (some are specific to Gensim/Word2Vec)
    ######
    dd_predictions = dict()
    for id in dd_mentions.keys():
        dd_predictions[id] = dict()
        dd_predictions[id]["pred_cui"] = []

    vocabSize = len(embeddings.key_to_index)
    sizeVST = embeddings.vector_size
    vocab = embeddings.key_to_index

    sizeVSO = len(dd_ref.keys())
    print("\tvocabSize:", vocabSize, "- embeddings size:", sizeVST, "- number of concepts:", sizeVSO)

    ######
    # Built mention embeddings from prediction set (same as for training set):
    ######
    print("\tMentions embeddings...")
    d_mentionVectors = dict()
    for id in dd_mentions.keys():
        d_mentionVectors[id] = numpy.zeros(sizeVST)
        l_tokens = dd_mentions[id]["mention"].split()
        for token in l_tokens:
            if token in vocab:
                d_mentionVectors[id] += (embeddings[token] / numpy.linalg.norm(embeddings[token]))
        d_mentionVectors[id] = d_mentionVectors[id] / len(l_tokens)
    print("\tDone.")

    ######
    # Regression prediction:
    ######
    print("\tRegression prediction...")
    Y_pred = numpy.zeros((len(dd_mentions.keys()), sizeVST))
    for i, id in enumerate(dd_mentions.keys()):
        x_test = numpy.zeros((1, sizeVST))
        x_test[0] = d_mentionVectors[id]
        Y_pred[i] = model.predict(x_test)[0]  # result of the regression for the i-th mention.
    del d_mentionVectors
    print("\tDone.")

    ######
    # Build labels/tags embeddings from ontology (same as training method):
    # (In fact, it could be directly reused from training to gain speed...)
    ######
    nbLabtags = 0
    dd_conceptVectors = dict()
    for cui in dd_ref.keys():
        dd_conceptVectors[cui] = dict()
        dd_conceptVectors[cui][dd_ref[cui]["label"]] = numpy.zeros(sizeVST)
        nbLabtags += 1
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                nbLabtags += 1
                dd_conceptVectors[cui][tag] = numpy.zeros(sizeVST)
    for cui in dd_ref.keys():
        l_tokens = dd_ref[cui]["label"].split()
        for token in l_tokens:
            if token in vocab:
                dd_conceptVectors[cui][dd_ref[cui]["label"]] += (
                        embeddings[token] / numpy.linalg.norm(embeddings[token]))
        if "tags" in dd_ref[cui].keys():
            for tag in dd_ref[cui]["tags"]:
                l_currentTagTokens = tag.split()
                for currentToken in l_currentTagTokens:
                    if currentToken in vocab:
                        dd_conceptVectors[cui][tag] += (embeddings[currentToken] / numpy.linalg.norm(embeddings[currentToken]))

    ######
    # Nearest neighbours calculation:
    ######
    labtagsVectorMatrix = numpy.zeros((nbLabtags, sizeVST))
    i = 0
    for cui in dd_conceptVectors.keys():
        for labtag in dd_conceptVectors[cui].keys():
            labtagsVectorMatrix[i] = dd_conceptVectors[cui][labtag]
            i += 1

    print('\tMatrix of distance calculation...')
    scoreMatrix = cdist(Y_pred, labtagsVectorMatrix, 'cosine')  # cdist() is an optimized algo to distance calculation.
    # (doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    # # Just to obtain the true cosine value
    # for i, id in enumerate(dd_mentions.keys()):
    #     j = -1
    #     for cui in dd_conceptVectors.keys():
    #         for labtag in dd_conceptVectors[cui].keys():
    #             j += 1
    #             scoreMatrix[i][j] = 1 - scoreMatrix[i][j]
    print("\tDone.")

    # For each mention, find back the nearest label/tag vector, then attribute the associated concept:
    for i, id in enumerate(dd_mentions.keys()):
        maximumScore = max(scoreMatrix[i])
        j = -1
        stopSearch = False
        for cui in dd_conceptVectors.keys():
            if stopSearch == True:
                break
            for labtag in dd_conceptVectors[cui].keys():
                j += 1
                if scoreMatrix[i][j] == maximumScore:
                    dd_predictions[id]["pred_cui"] = [cui]
                    stopSearch = True
                    break
    del dd_conceptVectors

    return dd_predictions


#######################################################################################################
# Test section
#######################################################################################################
if __name__ == '__main__':
    ################################################
    print("\nLOADING DATA:\n")
    ################################################

    print("loading OntoBiotope...")
    dd_obt = loader_ontobiotope("BB4/OntoBiotope_BioNLP-OST-2019.obo")
    print("loaded. (Nb of concepts in OBT =", len(dd_obt.keys()), ")")

    print("\nExtracting Bacterial Habitat hierarchy:")
    dd_habObt = select_subpart_hierarchy(dd_obt, 'OBT:000001')

    print("Done. (Nb of concepts in this subpart of OBT =", len(dd_habObt.keys()), ")")

    print("\nLoading BB4 corpora...")
    ddd_dataAll = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train", "BB4/BioNLP-OST-2019_BB-norm_dev",
                                       "BB4/BioNLP-OST-2019_BB-norm_test"])
    dd_habAll = extract_data(ddd_dataAll, l_type=["Habitat"])
    print("loaded.(Nb of mentions in whole corpus =", len(dd_habAll.keys()), ")")

    ddd_dataTrain = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train"])
    dd_habTrain = extract_data(ddd_dataTrain, l_type=["Habitat"])  # ["Habitat", "Phenotype", "Microorganism"]
    print("loaded.(Nb of mentions in train =", len(dd_habTrain.keys()), ")")

    ddd_dataDev = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habDev = extract_data(ddd_dataDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in dev =", len(dd_habDev.keys()), ")")

    ddd_dataTrainDev = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_train", "BB4/BioNLP-OST-2019_BB-norm_dev"])
    dd_habTrainDev = extract_data(ddd_dataTrainDev, l_type=["Habitat"])
    print("loaded.(Nb of mentions in train+dev =", len(dd_habTrainDev.keys()), ")")

    ddd_dataTest = loader_one_bb4_fold(["BB4/BioNLP-OST-2019_BB-norm_test"])
    dd_habTest = extract_data(ddd_dataTest, l_type=["Habitat"])
    print("loaded.(Nb of mentions in test =", len(dd_habTest.keys()), ")")

    print("\nLoading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format('./PubMed-w2v.bin', binary=True)
    print("Done...")

    ################################################
    print("\n\nPREPROCESSINGS:\n")
    ################################################

    print("Mentions lowercasing...")
    dd_BB4habTrain_lowercased = lowercaser_mentions(dd_habTrain)
    dd_BB4habDev_lowercased = lowercaser_mentions(dd_habDev)
    print("Done.\n")

    print("Lowercase references...")
    dd_habObt_lowercased = lowercaser_ref(dd_habObt)
    print("Done.")
    print(dd_BB4habTrain_lowercased)

    ################################################
    print("\n\nTRAINING and PREDICTING:\n")
    ################################################

    print("Training...")
    NNmodel = dense_layer_method_training(dd_BB4habTrain_lowercased, dd_habObt_lowercased, word_vectors)
    print("Training done.\n")

    print("Predicting...")
    dd_predictions = dense_layer_method_predicting(NNmodel, dd_BB4habDev_lowercased, dd_habObt_lowercased, word_vectors)
    print("Done.\n")

    print("Evaluating BB4 results on BB4 dev...")
    score_BB4_onDev = accuracy(dd_predictions, dd_habDev)
    print("score_BB4_onDev:", score_BB4_onDev)

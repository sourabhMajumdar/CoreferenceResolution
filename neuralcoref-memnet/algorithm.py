# coding: utf8
# cython: profile=True
# cython: infer_types=True
"""Coref resolution"""

from __future__ import unicode_literals
from __future__ import print_function

import sys
import glob
import os
import spacy
import numpy as np
import torch
from tqdm import tqdm

#from neuralcoref.utils import PACKAGE_DIRECTORY, SIZE_PAIR_IN, SIZE_SINGLE_IN
from utils import PACKAGE_DIRECTORY, SIZE_PAIR_IN, SIZE_SINGLE_IN
#from neuralcoref.compat import unicode_
from compat import unicode_
#from neuralcoref.document import Document, MENTION_TYPE, NO_COREF_LIST
#from document import Document, MENTION_TYPE, NO_COREF_LIST
from document import *
from utils import (encode_distance, BATCH_SIZE_PATH, SIZE_FP,
                               SIZE_FP_COMPRESSED, SIZE_FS, SIZE_FS_COMPRESSED,
                               SIZE_GENRE, SIZE_PAIR_IN, SIZE_SINGLE_IN,SIZE_EMBEDDING)
from dataset import (NCDataset, NCBatchSampler,
    load_embeddings_from_file, padder_collate,
    SIZE_PAIR_IN, SIZE_SINGLE_IN, SIZE_EMBEDDING)

from conllparser import *
from model import Model


#######################
##### UTILITIES #######

MAX_FOLLOW_UP = 50
FEATURES_NAMES = ["mentions_features",          # 0
                  "mentions_labels",            # 1
                  "mentions_pairs_length",      # 2
                  "mentions_pairs_start_index", # 3
                  "mentions_spans",             # 4
                  "mentions_words",             # 5
                  "pairs_ant_index",            # 6
                  "pairs_features",             # 7
                  "pairs_labels",               # 8
                  "mentions_stories",               # 9
                  ]
#######################
###### CLASSES ########
# copy this file but, with your pre-trained model which is stored inside the /checkpoints file
#class Model(object):
#    '''
#    Coreference neural model
#    '''
#    def __init__(self, model_path):
#        weights, biases = [], []
#        for file in sorted(os.listdir(model_path)):
#            if file.startswith("single_mention_weights"):
#                w = np.load(os.path.join(model_path, file))
#                weights.append(w)
#            if file.startswith("single_mention_bias"):
#                w = np.load(os.path.join(model_path, file))
#                biases.append(w)
#        self.single_mention_model = list(zip(weights, biases))
#        weights, biases = [], []
#        for file in sorted(os.listdir(model_path)):
#            if file.startswith("pair_mentions_weights"):
#                w = np.load(os.path.join(model_path, file))
#                weights.append(w)
#            if file.startswith("pair_mentions_bias"):
#                w = np.load(os.path.join(model_path, file))
#                biases.append(w)
#        self.pair_mentions_model = list(zip(weights, biases))
#
#    def _score(self, features, layers):
#        layer_count = 0
#        for weights, bias in layers:
#            layer_count += 1
#            print("layer# ",layer_count)
#            print("features", features.shape)
#            print("weights",weights.shape)
#            features = np.matmul(weights, features) + bias
#            if weights.shape[0] > 1:
#                features = np.maximum(features, 0) # ReLU
#        return np.sum(features, axis=0)
#
#    def get_multiple_single_score(self, first_layer_input):
#        return self._score(first_layer_input, self.single_mention_model)
#
#    def get_multiple_pair_score(self, first_layer_input):
#        return self._score(first_layer_input, self.pair_mentions_model)
#
class CModel(object):
    '''
    Coreference neural model
    '''
    def __init__(self, model_path):

        list_of_files = glob.glob(model_path + '*modelranking') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)

        checkpoint_file = latest_file
        embed_path = 'weights/'
        tensor_embeddings, voc = load_embeddings_from_file(embed_path + "tuned_word")
        print("🏝 Build model")
        h1 = 1000
        h2 = 500
        h3 = 500
        self.model = Model(len(voc), SIZE_EMBEDDING, h1,h2,h3, SIZE_PAIR_IN, SIZE_SINGLE_IN)
        self.model.load_embeddings(tensor_embeddings)
        cuda = torch.cuda.is_available()

        if cuda:
            model.cuda()
        if checkpoint_file is not None:
            print("⛄️ Loading model from", checkpoint_file)
            self.model.load_state_dict(torch.load(checkpoint_file) if cuda else torch.load(checkpoint_file, map_location=lambda storage, loc: storage))
            self.model.eval()

    #def _score(self, features,model):
    #    print(SIZE_SINGLE_IN)
    #    print(features.shape)    
    #    print(type(features))    
    #    return self.model(features)

    def get_multiple_single_score(self, first_layer_input):
        return self.model(first_layer_input)

    def get_multiple_pair_score(self, first_layer_input):
        return self.model(first_layer_input)


class Coref(object):
    '''
    Main coreference resolution algorithm
    '''
    def __init__(self, nlp=None,greedyness=0.5, max_dist=50, max_dist_match=500, conll=None,
                 blacklist=True, debug=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        embed_path = 'weights/'
        if embed_path is not None:
            self.embed_extractor = EmbeddingExtractor(embed_path)
        #model_path = os.path.join(PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/")
        #model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        model_path = "checkpoints/"
        print("Loading neuralcoref model from", model_path)
        self.coref_model = CModel(model_path)
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info('en_core_web_sm')
                model = 'en_core_web_sm'
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                spacy.info('en')
                model = 'en'
            nlp = spacy.load(model)
        self.data = Document(nlp, conll=conll, blacklist=blacklist, model_path='weights/')
        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}

    ###################################
    #### ENTITY CLUSTERS FUNCTIONS ####
    ###################################

    def get_single_mention_features_conll(self, mention, compressed=True):
        ''' Compressed or not single mention features'''
        if not compressed:
            _, features = self.get_single_mention_features(mention)
            return features[np.newaxis, :]
        feat_l = [mention.features_["01_MentionType"],
                  mention.features_["02_MentionLength"],
                  mention.index,
                  len(self.data.mentions),
                  mention.features_["04_IsMentionNested"],
                  #self.data.mention.genre_,
                  0
                 ]
        return feat_l

    def get_pair_mentions_features_conll(self, m1, m2, compressed=True):
        ''' Compressed or not single mention features'''
        if not compressed:
            _, features = self.data.get_pair_mentions_features(m1, m2)
            return features[np.newaxis, :]
        features_, _ = self.data.get_pair_mentions_features(m1, m2)
        feat_l = [features_["00_SameSpeaker"],
                  features_["01_AntMatchMentionSpeaker"],
                  features_["02_MentionMatchSpeaker"],
                  features_["03_HeadsAgree"],
                  features_["04_ExactStringMatch"],
                  features_["05_RelaxedStringMatch"],
                  features_["06_SentenceDistance"],
                  features_["07_MentionDistance"],
                  features_["08_Overlapping"],
                 ]
        return feat_l

    def _prepare_clusters(self):
        '''
        Clean up and prepare one cluster for each mention
        '''
        self.mention_to_cluster = list(range(len(self.data.mentions)))
        self.clusters = dict((i, [i]) for i in self.mention_to_cluster)
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}
        for mention in self.mention_to_cluster:
            self.mentions_single_scores[mention] = None
            self.mentions_pairs_scores[mention] = {}

    def _merge_coreference_clusters(self, ant_idx, mention_idx):
        '''
        Merge two clusters together
        '''
        if self.mention_to_cluster[ant_idx] == self.mention_to_cluster[mention_idx]:
            return

        remove_id = self.mention_to_cluster[ant_idx]
        keep_id = self.mention_to_cluster[mention_idx]
        for idx in self.clusters[remove_id]:
            self.mention_to_cluster[idx] = keep_id
            self.clusters[keep_id].append(idx)

        del self.clusters[remove_id]

    def remove_singletons_clusters(self):
        remove_id = []
        for key, mentions in self.clusters.items():
            if len(mentions) == 1:
                remove_id.append(key)
                self.mention_to_cluster[key] = None
        for rem in remove_id:
            del self.clusters[rem]

    def display_clusters(self):
        '''
        Print clusters informations
        '''
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print("cluster", key, "(", ", ".join(unicode_(self.data[m]) for m in mentions), ")")

    ###################################
    ####### MAIN COREF FUNCTIONS ######
    ###################################

    def run_coref_on_mentions_OLD(self, mentions):
        '''
        Run the coreference model on a mentions list
        '''
        best_ant = {}
        best_score = {}
        n_ant = 0
        print(mentions)
        sardarji = input("PRINTED MENTIONS EXTRACTED")
        inp = np.empty((SIZE_SINGLE_IN, len(mentions)))
        print("SHAPE OF INP,",inp.shape)
        hua = input()
        for i, mention_idx in enumerate(mentions):
            mention = self.data[mention_idx]
            print(mention)
            frio = input("mention extraced from data")
            print(type(mention))
            shah = input("mention type")
            print()
            print("mention embedding", mention.embedding.shape)
            print("mention.features",mention.features.shape)
            print("data.genre.shape,",self.data.genre.shape)
            inp[:len(mention.embedding), i] = mention.embedding
            inp[:len(mention.embedding), i] = mention.features
            inp[:len(mention.embedding), i] = self.data.genre

        score = self.coref_model.get_multiple_single_score(inp.T)
        print("SINGLE SCORES CALCULATED")
        cenkls = input()
        #score = self.coref_model(tuple(mentions_spans,mentions_words,mentions_features))
        #score = self.coref_model.get_single(single_inputs)
        for mention_idx, s in zip(mentions, score):
            self.mentions_single_scores[mention_idx] = s
            best_score[mention_idx] = s - 50 * (self.greedyness - 0.5)

        for mention_idx, ant_list in self.data.get_candidate_pairs(mentions, self.max_dist, self.max_dist_match):
            if len(ant_list) == 0:
                continue
            inp_l = []
            for ant_idx in ant_list:
                mention = self.data[mention_idx]
                antecedent = self.data[ant_idx]
                feats_, pwf = self.data.get_pair_mentions_features(antecedent, mention)
                inp_l.append(pwf)
            inp = np.stack(inp_l, axis=0)
            #score = self.coref_model.get_multiple_pair_score(inp.T)
            # spans, words, single_features, ant_spans, ant_words, ana_spans, ana_words, pair_features
            #score = self.coref_model.get_multiple_pair_score(tuple())
            score = self.coref_model(pairs_inputs)
            for ant_idx, s in zip(ant_list, score):
                self.mentions_pairs_scores[mention_idx][ant_idx] = s
                if s > best_score[mention_idx]:
                    best_score[mention_idx] = s
                    best_ant[mention_idx] = ant_idx
            if mention_idx in best_ant:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[mention_idx], mention_idx)
        return (n_ant, best_ant)

    def run_coref_on_mentions(self, mentions):
        '''
        Run the coreference model on a mentions list
        '''
        best_ant = {}
        best_score = {}
        n_ant = 0
        #print(mentions)
        #inp = np.empty((SIZE_SINGLE_IN, len(mentions)))
        #print("SHAPE OF INP SHAPE")
        #print(inp.shape)
        #yur = input()
        #for i, mention_idx in enumerate(mentions):
        #    mention = self.data[mention_idx]
        #    print(mention)
        #    frio = input("mention extraced from data")
        #    print(type(mention))
        #    shah = input("mention type")
        #    print()
        #    print("mention embedding", mention.embedding.shape)
        #    inp[:len(mention.embedding), i] = mention.embedding
        #    inp[:len(mention.embedding), i] = mention.features
        #    inp[:len(mention.embedding), i] = self.data.genre

        mention_idx_list = []
        mentions_spans = []
        mentions_words = []
        mentions_features = []
        pairs_ant_idx = []
        pairs_features = []
        pairs_labels = []
        mentions_labels = []
        mentions_pairs_start = []
        mentions_pairs_length = []
        mentions_location = []

        mentions_stories = []
        n_mentions = 0
        total_pairs = 0

        #print(mentions)
        #oven_sotry = input('MENTIONS PRINTED')
        #if debug: print("mentions", self.mentions, str([m.gold_label for m in self.mentions]))
        # create 2 for loops, one for single pairs and one for pairs

        for mention_idx, antecedents_idx in list(self.data.get_candidate_pairs(mentions, self.max_dist, self.max_dist_match)):
            n_mentions += 1
            doc_id = 1
            mention = self.data[mention_idx]

            # let's create the story
            story_embeds = []
            raw_utterances = self.get_utterances()
            for utt_index in range(mention.utterance_index) :
                utt_dealt = raw_utterances[utt_index]
                for token in utt_dealt :
                    # since mention_words_idx works on Mention, we convert every token into a mention
                    token_word_idx = word_idx_finder(self.embed_extractor,token.text)
                    #story_embeds.append(token_embed.tolist())
                    story_embeds.append(token_word_idx)
            final_utt_dealt = raw_utterances[mention.utterance_index]
            for token_index in range(mention.start) :
                token_word_idx = word_idx_finder(self.embed_extractor,final_utt_dealt[token_index].text)
                #story_embeds.append(token_embed.tolist())
                story_embeds.append(token_word_idx)

            mentions_stories.append(story_embeds)
            mention_idx_list.append(mention_idx)
            mentions_spans.append(mention.spans_embeddings)
            w_idx = mention_words_idx(self.embed_extractor, mention)

            if w_idx is None:
                print("error in", self.name, self.part, mention.utterance_index)
            mentions_words.append(w_idx)
            mentions_features.append(self.get_single_mention_features_conll(mention))
            mentions_location.append([mention.start, mention.end, mention.utterance_index, mention_idx, doc_id])
            ants = [self.data.mentions[ant_idx] for ant_idx in antecedents_idx]

            # Some display functions
            #tuy = input()
            #print("************************************************************************************************")
            #print("MENTION IDX,",mention_idx)
            #print("MENTION REFRED,",mention)
            #print("MENTION SPANS,",mention.spans_embeddings.shape)
            #print("MENTION FEATURES,",self.get_single_mention_features_conll(mention))
            #print("MENTION LOCATION,",[mention.start,mention.end,mention.utterance_index,mention_idx,doc_id])
            #print("ANTS ,",ants)
            #print("*************************************************************************************************")
            #hua = input()
            no_antecedent = not any(ant.gold_label == mention.gold_label for ant in ants) or mention.gold_label is None
            if antecedents_idx:
                pairs_ant_idx += [idx for idx in antecedents_idx]
                pairs_features += [self.get_pair_mentions_features_conll(ant, mention) for ant in ants]
                ant_labels = [0 for ant in ants] if no_antecedent else [1 if ant.gold_label == mention.gold_label else 0 for ant in ants]
                pairs_labels += ant_labels
            mentions_labels.append(1 if no_antecedent else 0)
            mentions_pairs_start.append(total_pairs)
            total_pairs += len(ants)
            mentions_pairs_length.append(len(ants))

        
        out_dict = {FEATURES_NAMES[0]: mentions_features,
                    FEATURES_NAMES[1]: mentions_labels,
                    FEATURES_NAMES[2]: mentions_pairs_length,
                    FEATURES_NAMES[3]: mentions_pairs_start,
                    FEATURES_NAMES[4]: mentions_spans,
                    FEATURES_NAMES[5]: mentions_words,
                    #FEATURES_NAMES[6]: pairs_ant_idx if pairs_ant_idx else None,
                    FEATURES_NAMES[6]: pairs_ant_idx if pairs_ant_idx else list(),
                    #FEATURES_NAMES[7]: pairs_features if pairs_features else None,
                    FEATURES_NAMES[7]: pairs_features if pairs_features else list(),
                    #FEATURES_NAMES[8]: pairs_labels if pairs_labels else None,
                    FEATURES_NAMES[8]: pairs_labels if pairs_labels else list(),
                    FEATURES_NAMES[9] : mentions_stories
                    }
        gathering_dict = dict((feat, None) for feat in FEATURES_NAMES)
        n_mentions_list = []
        pairs_ant_index = 0
        pairs_start_index = 0
        for n, p, arrays_dict in tqdm([(n_mentions,total_pairs,out_dict)]):
            #print(out_dict)
            #pizza_hut = input('OUT DICT PRINTED')
            #print(arrays_dict)
            #dominoes = input('ARRAYS DICT PRINTED')
            for f in FEATURES_NAMES:
                if gathering_dict[f] is None:
                    gathering_dict[f] = arrays_dict[f]
                else:
                    if f == FEATURES_NAMES[6]:
                        array = [a + pairs_ant_index for a in arrays_dict[f]]
                    elif f == FEATURES_NAMES[3]:
                        array = [a + pairs_start_index for a in arrays_dict[f]]
                    else:
                        array = arrays_dict[f]
                    gathering_dict[f] += array
            pairs_ant_index += n
            pairs_start_index += p
            n_mentions_list.append(n)

        mention_feature_dict = dict()
        pairs_feature_dict = dict()
        train_phase = True

        for feature in FEATURES_NAMES[:10]:
            print("Building numpy array for", feature, "length", len(gathering_dict[feature]))
            if feature != "mentions_spans":
                #array = np.array(gathering_dict[feature])
                # check if we are dealing with length of memories
                if feature == "mentions_stories" or feature == "pairs_stories" : 
                    gathering_array = []
                    max_story_len = 200
                    for story in gathering_dict[feature] :
                        #print(len(story[0]))
                        #print(len(story[1]))
                        #random_pause = input()
                        if len(story) > 200 :
                            final_story = story[-200:]
                        else :
                            number_to_append = max(0,max_story_len - len(story))
                            #number_to_append = min(number_to_append,50)
                            final_story = story + number_to_append*[0]
                            #print(final_story)
                            #print(len(final_story))
                            #random_pause = input()
                        gathering_array.append(final_story)
                    array = np.array(gathering_array)
                    print(array.shape)
                else :
                    array = np.array(gathering_dict[feature])

                if array.ndim == 1:
                    print("expand_dims for feature, ",feature)
                    array = np.expand_dims(array, axis=1)
            else:
                array = np.stack(gathering_dict[feature])
            # check_numpy_array(feature, array, n_mentions_list)
            print("Saving numpy", feature, "size", array.shape)
            #array_save = input()
            if feature.startswith("mentions") :
                mention_feature_dict[feature] = array
            if feature.startswith("pairs") :
                pairs_feature_dict[feature] = array

        # zip it with pairs dict
        self.mentions = list(zip(*(arr for key, arr in sorted(mention_feature_dict.items()))))
        self.pairs = list(zip(*(arr for key, arr in sorted(pairs_feature_dict.items()))))
        #print("LEN OF PAIRS IS,",len(self.pairs))

        #jsk = input("PRINTING THE PAIRS")
        #print(self.pairs)
        #sdghr = input("ALL PAIRS PRINTED")
        #print("MENTION PAIRS LENGTH IS,",mention_feature_dict['mentions_pairs_length'])
        #victoria = input()

        for i in range(len(mention_feature_dict[FEATURES_NAMES[0]])) :
            mention_idx = mention_idx_list[i]
            features_raw = mention_feature_dict['mentions_features'][i,:]
            #print("FEATUERES_RAW_PRINTED_is,",features_raw)
            label = mention_feature_dict['mentions_labels'][i,:]
            pairs_length = mention_feature_dict['mentions_pairs_length'][i,:]
            pairs_start_index = mention_feature_dict['mentions_pairs_start_index'][i]
            mentions_stories = mention_feature_dict['mentions_stories'][i]



            spans = mention_feature_dict['mentions_spans'][i,:]
            words = mention_feature_dict['mentions_words'][i,:]

            pairs_start_index = np.asscalar(pairs_start_index)
            pairs_length = np.asscalar(pairs_length)

            # Build features array (float) from raw features (int)
            assert features_raw.shape[0] == SIZE_FS_COMPRESSED
            features = np.zeros((SIZE_FS,))
            features[features_raw[0]] = 1
            features[4:15] = encode_distance(features_raw[1])
            features[15] = features_raw[2].astype(float) / features_raw[3].astype(float)
            features[16] = features_raw[4]
            features[features_raw[5] + 17] = 1


            #print("====================================<>============================================")
            #print("TYPE OF SPANS,",type(spans))
            #print("TYPE OF WORDS,",type(words))
            #print("TYPE OF FEATURES,",type(features))
            #print("====================================<>============================================")


            spans = spans[np.newaxis,:]
            print("PRINTING SHAPE OF WORDS")
            print(words.shape)
            words = words[np.newaxis,:]
            features = features[np.newaxis,:]
            mentions_stories = mentions_stories[np.newaxis,:]

            spans = torch.from_numpy(spans).float()
            words = torch.from_numpy(words)
            features = torch.from_numpy(features).float()
            mentions_stories = torch.from_numpy(mentions_stories)
            #print(mentions_stories.size())
            #print(words.size())
            #kake = input("size of mentions stories is ")

            # inputs for the single mentions

            #print("SINGLE SCORES COMPUTING")
            single_inputs = (spans, words, features,mentions_stories)
            score = self.coref_model.get_multiple_single_score(single_inputs).tolist()[0][0]

            #print("PRINTING SINGLE SCORE")
            #print(score)
            #sgbet = input("SINGLE SCORE PRINTED")
            self.mentions_single_scores[mention_idx] = score
            best_score[mention_idx] = score - 50 * (self.greedyness - 0.5)
            #print("SINGLE SCORES COMPUTED")

            if pairs_length==0 :
                continue

            start = pairs_start_index
            end = pairs_start_index + pairs_length
            pairs = self.pairs[start:end]
            #print("START IS,",start)
            #print("END IS,",end)
            #print("PAIRS LENGTH,",pairs_length)
            #print("LEN OF PAIRS IS,",len(pairs))
            assert len(pairs) == pairs_length
            assert len(pairs[0]) == 3 # pair[i] = (pairs_ant_index, pairs_features, pairs_labels)
            pairs_ant_index, pairs_features_raw, pairs_labels = list(zip(*pairs))

            pairs_features_raw = np.stack(pairs_features_raw)
            pairs_labels = np.squeeze(np.stack(pairs_labels), axis=1)

            # Build pair features array (float) from raw features (int)
            assert pairs_features_raw[0, :].shape[0] == SIZE_FP_COMPRESSED
            pairs_features = np.zeros((len(pairs_ant_index), SIZE_FP))
            pairs_features[:, 0:6] = pairs_features_raw[:, 0:6]
            pairs_features[:, 6:17] = encode_distance(pairs_features_raw[:, 6])
            pairs_features[:, 17:28] = encode_distance(pairs_features_raw[:, 7])
            pairs_features[:, 28] = pairs_features_raw[:, 8]
            # prepare antecent features

            # printing antecedent features
            #hsya = input("PRINTING DATA MENTIONS")
            #print(self.data.mentions)
            #uba = input("PRINTED DATA MENTIONS")
            ant_features_raw = np.concatenate([self.mentions[np.asscalar(idx)][0][np.newaxis, :] for idx in pairs_ant_index])
            ant_features = np.zeros((pairs_length, SIZE_FS-SIZE_GENRE))
            ant_features[:, ant_features_raw[:, 0]] = 1
            ant_features[:, 4:15] = encode_distance(ant_features_raw[:, 1])
            ant_features[:, 15] = ant_features_raw[:, 2].astype(float) / ant_features_raw[:, 3].astype(float)
            ant_features[:, 16] = ant_features_raw[:, 4]
            pairs_features[:, 29:46] = ant_features
            # Here we keep the genre 
            ana_features = np.tile(features, (pairs_length, 1))
            pairs_features[:, 46:] = ana_features

            ant_spans = np.concatenate([self.mentions[np.asscalar(idx)][4][np.newaxis, :] for idx in pairs_ant_index])
            ant_words = np.concatenate([self.mentions[np.asscalar(idx)][6][np.newaxis, :] for idx in pairs_ant_index])
            ana_spans = np.tile(spans, (pairs_length, 1))
            ana_words = np.tile(words, (pairs_length, 1))

            ant_spans = ant_spans[np.newaxis,:]
            ant_words = ant_words[np.newaxis,:]
            ana_spans = ana_spans[np.newaxis,:]
            ana_words = ana_words[np.newaxis,:]
            pairs_features = pairs_features[np.newaxis,:]

            ant_spans = torch.from_numpy(ant_spans).float()
            ant_words = torch.from_numpy(ant_words)
            ana_spans = torch.from_numpy(ana_spans).float()
            ana_words = torch.from_numpy(ana_words)
            pairs_features = torch.from_numpy(pairs_features).float()

            labels_stack = np.concatenate((pairs_labels, label), axis=0)
            assert labels_stack.shape == (pairs_length + 1,)
            labels = torch.from_numpy(labels_stack).float()

            # inputs for the pairs of mentions
            pairs_inputs = (spans, words, features,ant_spans, ant_words,ana_spans, ana_words,pairs_features,mentions_stories)


            #print("PAIRS INPUT CREATED")

            score = self.coref_model.get_multiple_pair_score(pairs_inputs).tolist()[0][:-1]
            #print("SCORES GOT IS ")
            #print(score)
            #hutys = input("SCORES PRINTED")
            for ant_idx, s in zip(pairs_ant_idx, score):
                self.mentions_pairs_scores[mention_idx][ant_idx] = s
                if s > best_score[mention_idx]:
                    best_score[mention_idx] = s
                    best_ant[mention_idx] = ant_idx
            if mention_idx in best_ant:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[mention_idx], mention_idx)

        #score = self.coref_model.get_multiple_single_score(inp.T)
        #score = self.coref_model(tuple(mentions_spans,mentions_words,mentions_features
            
            

        # find a way to access the mention_idx
        return (n_ant, best_ant)

    def run_coref_on_utterances(self, last_utterances_added=False, follow_chains=True, debug=False):
        ''' Run the coreference model on some utterances

        Arg:
            last_utterances_added: run the coreference model over the last utterances added to the data
            follow_chains: follow coreference chains over previous utterances
        '''
        if debug: print("== run_coref_on_utterances == start")
        self._prepare_clusters()
        if debug: self.display_clusters()
        #print(last_utterances_added)
        #keventers = input("we are seeing last utterances")
        mentions = list(self.data.get_candidate_mentions(last_utterances_added=last_utterances_added))
        #print(mentions)
        #bowl_story = input('EXECUTED GET CANDIDATE MENTIONS AND PRINTING MENTIONS')
        n_ant, antecedents = self.run_coref_on_mentions(mentions)
        mentions = antecedents.values()
        if follow_chains and last_utterances_added and n_ant > 0:
            i = 0
            while i < MAX_FOLLOW_UP:
                i += 1
                n_ant, antecedents = self.run_coref_on_mentions(mentions)
                mentions = antecedents.values()
                if n_ant == 0:
                    break
        if debug: self.display_clusters()
        if debug: print("== run_coref_on_utterances == end")

    def one_shot_coref(self, utterances, utterances_speakers_id=None, context=None,
                       context_speakers_id=None, speakers_names=None):
        ''' Clear history, load a list of utterances and an optional context and run the coreference model on them

        Arg:
        - `utterances` : iterator or list of string corresponding to successive utterances (in a dialogue) or sentences.
            Can be a single string for non-dialogue text.
        - `utterances_speakers_id=None` : iterator or list of speaker id for each utterance (in the case of a dialogue).
            - if not provided, assume two speakers speaking alternatively.
            - if utterances and utterances_speaker are not of the same length padded with None
        - `context=None` : iterator or list of string corresponding to additionnal utterances/sentences sent prior to `utterances`. Coreferences are not computed for the mentions identified in `context`. The mentions in `context` are only used as possible antecedents to mentions in `uterrance`. Reduce the computations when we are only interested in resolving coreference in the last sentences/utterances.
        - `context_speakers_id=None` : same as `utterances_speakers_id` for `context`. 
        - `speakers_names=None` : dictionnary of list of acceptable speaker names (strings) for speaker_id in `utterances_speakers_id` and `context_speakers_id`
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        return self.get_clusters()

    def one_shot_coref_NEW(self, utterances, utterances_speakers_id=None, context=None,
                       context_speakers_id=None, speakers_names=None):
        ''' Clear history, load a list of utterances and an optional context and run the coreference model on them

        Arg:
        - `utterances` : iterator or list of string corresponding to successive utterances (in a dialogue) or sentences.
            Can be a single string for non-dialogue text.
        - `utterances_speakers_id=None` : iterator or list of speaker id for each utterance (in the case of a dialogue).
            - if not provided, assume two speakers speaking alternatively.
            - if utterances and utterances_speaker are not of the same length padded with None
        - `context=None` : iterator or list of string corresponding to additionnal utterances/sentences sent prior to `utterances`. Coreferences are not computed for the mentions identified in `context`. The mentions in `context` are only used as possible antecedents to mentions in `uterrance`. Reduce the computations when we are only interested in resolving coreference in the last sentences/utterances.
        - `context_speakers_id=None` : same as `utterances_speakers_id` for `context`. 
        - `speakers_names=None` : dictionnary of list of acceptable speaker names (strings) for speaker_id in `utterances_speakers_id` and `context_speakers_id`
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.set_utterances(context, context_speakers_id, speakers_names)
        self.continuous_coref(utterances, utterances_speakers_id, speakers_names)
        clusters, mention_to_cluster = self.get_clusters()
        actual_clusters = dict()
        for key, values in clusters.items() :
            if self.data.mentions[key] in actual_clusters.keys() :
                list_of_entities = actual_clusters[key]
            else :
                list_of_entities = list()
            for mention_idx in values :
                list_of_entities.append(self.data.mentions[mention_idx])
            actual_clusters[self.data.mentions[key]] = list_of_entities
        return actual_clusters
        #return self.get_clusters()

    def continuous_coref(self, utterances, utterances_speakers_id=None, speakers_names=None):
        '''
        Only resolve coreferences for the mentions in the utterances
        (but use the mentions in previously loaded utterances as possible antecedents)
        Arg:
            utterances : iterator or list of string corresponding to successive utterances
            utterances_speaker : iterator or list of speaker id for each utterance.
                If not provided, assume two speakers speaking alternatively.
                if utterances and utterances_speaker are not of the same length padded with None
            speakers_names : dictionnary of list of acceptable speaker names for each speaker id
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.add_utterances(utterances, utterances_speakers_id, speakers_names)
        self.run_coref_on_utterances(last_utterances_added=True, follow_chains=True)
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_utterances(self, last_utterances_added=True):
        ''' Retrieve the list of parsed uterrances'''
        if last_utterances_added and len(self.data.last_utterances_loaded):
            return [self.data.utterances[idx] for idx in self.data.last_utterances_loaded]
        else:
            return self.data.utterances

    def get_resolved_utterances(self, last_utterances_added=True, blacklist=True):
        ''' Return a list of utterrances text where the '''
        coreferences = self.get_most_representative(last_utterances_added, blacklist)
        resolved_utterances = []
        for utt in self.get_utterances(last_utterances_added=last_utterances_added):
            resolved_utt = ""
            in_coref = None
            for token in utt:
                if in_coref is None:
                    for coref_original, coref_replace in coreferences.items():
                        if coref_original[0] == token:
                            in_coref = coref_original
                            resolved_utt += coref_replace.text.lower()
                            break
                    if in_coref is None:
                        resolved_utt += token.text_with_ws
                if in_coref is not None and token == in_coref[-1]:
                    resolved_utt += ' ' if token.whitespace_ and resolved_utt[-1] is not ' ' else ''
                    in_coref = None
            resolved_utterances.append(resolved_utt)
        return resolved_utterances

    def get_mentions(self):
        ''' Retrieve the list of mentions'''
        return self.data.mentions

    def get_scores(self):
        ''' Retrieve scores for single mentions and pair of mentions'''
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=False, blacklist=False):
        ''' Retrieve cleaned clusters'''
        clusters = self.clusters
        mention_to_cluster = self.mention_to_cluster
        remove_id = []
        if blacklist:
            for key, mentions in clusters.items():
                cleaned_list = []
                for mention_idx in mentions:
                    mention = self.data.mentions[mention_idx]
                    if mention.lower_ not in NO_COREF_LIST:
                        cleaned_list.append(mention_idx)
                clusters[key] = cleaned_list
            # Also clean up keys so we can build coref chains in self.get_most_representative
            added = {}
            for key, mentions in clusters.items():
                if self.data.mentions[key].lower_ in NO_COREF_LIST:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
                    if mentions:
                        added[mentions[0]] = mentions
            for rem in remove_id:
                del clusters[rem]
            clusters.update(added)

        if remove_singletons:
            remove_id = []
            for key, mentions in clusters.items():
                if len(mentions) == 1:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
            for rem in remove_id:
                del clusters[rem]

        return clusters, mention_to_cluster

    def get_most_representative(self, last_utterances_added=True, blacklist=True):
        '''
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        '''
        clusters, _ = self.get_clusters(remove_singletons=True, blacklist=blacklist)
        coreferences = {}
        for key in self.data.get_candidate_mentions(last_utterances_added=last_utterances_added):
            if self.mention_to_cluster[key] is None:
                continue
            mentions = clusters.get(self.mention_to_cluster[key], None)
            if mentions is None:
                continue
            representative = self.data.mentions[key]
            for mention_idx in mentions[1:]:
                mention = self.data.mentions[mention_idx]
                if mention.mention_type is not representative.mention_type:
                    if mention.mention_type == MENTION_TYPE["PROPER"] \
                        or (mention.mention_type == MENTION_TYPE["NOMINAL"] and
                                representative.mention_type == MENTION_TYPE["PRONOMINAL"]):
                        coreferences[self.data.mentions[key]] = mention
                        representative = mention

        return coreferences

# coding: utf8
"""Conll parser"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import re
import sys
import codecs
import argparse
import time
import os
import io
import pickle

import spacy
from spacy.tokens import Doc

import numpy as np

from tqdm import tqdm

#from .compat import unicode_
from compat import unicode_
#from .document import Mention, Document, Speaker, EmbeddingExtractor, MISSING_WORD
#from document import Mention, Document, Speaker, EmbeddingExtractor, MISSING_WORD
from document import *
#from .utils import parallel_process
from utils import parallel_process

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REMOVED_CHAR = ["/", "%", "*"]
NORMALIZE_DICT = {"/.": ".",
				  "/?": "?",
				  "-LRB-": "(",
				  "-RRB-": ")",
				  "-LCB-": "{",
				  "-RCB-": "}",
				  "-LSB-": "[",
				  "-RSB-": "]"}

CONLL_GENRES = {"bc": 0, "bn": 1, "mz": 2, "nw": 3, "pt": 4, "tc": 5, "wb": 6}

FEATURES_NAMES = ["mentions_features",          # 0
				  "mentions_labels",            # 1
				  "mentions_pairs_length",      # 2
				  "mentions_pairs_start_index", # 3
				  "mentions_spans",             # 4
				  "mentions_words",             # 5
				  "pairs_ant_index",            # 6
				  "pairs_features",             # 7
				  "pairs_labels",               # 8
				  "mentions_stories",            # 9
				  "locations",                  # 10
				  "conll_tokens",               # 11
				  "spacy_lookup",               # 12
				  "doc",                        # 13
				  ]

MISSED_MENTIONS_FILE = os.path.join(PACKAGE_DIRECTORY, "test_mentions_identification.txt")
SENTENCES_PATH = os.path.join(PACKAGE_DIRECTORY, "test_sentences.txt")

###################
### UTILITIES #####

def clean_token(token):
	cleaned_token = token
	if cleaned_token in NORMALIZE_DICT:
		cleaned_token = NORMALIZE_DICT[cleaned_token]
	if cleaned_token not in REMOVED_CHAR:
		for char in REMOVED_CHAR:
			cleaned_token = cleaned_token.replace(char, u'')
	if len(cleaned_token) == 0:
		cleaned_token = ","
	return cleaned_token

def mention_words_idx(embed_extractor, mention, debug=False):
	# index of the word in the tuned embeddings no need for normalizing,
	# it is already performed in set_mentions_features()
	# We take them in the tuned vocabulary which is a smaller voc tailored from conll
	words = []
	#mention_word_idx_description = input("Inside the mention_words_idx function\n finds the index of the word in the tuned embeddings no need for normalizing\n it is already performed in set_mentions_features()\n We take them in the tuned vocabulary which is a smaller voc tailored from conll")
	for _, w in sorted(mention.words_embeddings_.items()):
		if w not in embed_extractor.tun_idx:
			if debug:
				print("No matching tokens in tuned voc for word ", w, "surrounding or inside mention", mention)
				#mention_words_debug = input()
			words.append(MISSING_WORD)
		else:
			words.append(w)
	return [embed_extractor.tun_idx[w] for w in words]

def word_idx_finder(embed_extractor,word=MISSING_WORD) :
	if word in embed_extractor.tun_idx :
		return embed_extractor.tun_idx[word]
	else :
		return embed_extractor.tun_idx[MISSING_WORD]

def check_numpy_array(feature, array, n_mentions_list, compressed=True):
	for n_mentions in n_mentions_list:
		if feature == FEATURES_NAMES[0]:
			assert array.shape[0] == len(n_mentions)
			if compressed:
				assert np.array_equiv(array[:, 3], np.array([len(n_mentions)] * len(n_mentions)))
				assert np.max(array[:, 2]) == len(n_mentions)-1
				assert np.min(array[:, 2]) == 0
		elif feature == FEATURES_NAMES[1]:
			assert array.shape[0] == len(n_mentions)
		elif feature == FEATURES_NAMES[2]:
			assert array.shape[0] == len(n_mentions)
			assert np.array_equiv(array[:, 0], np.array(list(range(len(n_mentions)))))
		elif feature == FEATURES_NAMES[3]:
			assert array.shape[0] == len(n_mentions)
			assert np.array_equiv(array[:, 0], np.array([p*(p-1)/2 for p in range(len(n_mentions))]))
		elif feature == FEATURES_NAMES[4]:
			assert array.shape[0] == len(n_mentions)
		elif feature == FEATURES_NAMES[5]:
			assert array.shape[0] == len(n_mentions)
		elif feature == FEATURES_NAMES[6]:
			assert array.shape[0] == len(n_mentions)*(len(n_mentions)-1)/2
			assert np.max(array) == len(n_mentions)-2
		elif feature == FEATURES_NAMES[7]:
			if compressed:
				assert array.shape[0] == len(n_mentions)*(len(n_mentions)-1)/2
				assert np.max(array[:, 7]) == len(n_mentions)-2
				assert np.min(array[:, 7]) == 0
		elif feature == FEATURES_NAMES[8]:
			assert array.shape[0] == len(n_mentions)*(len(n_mentions)-1)/2

###############################################################################################
### PARALLEL FCT (has to be at top-level of the module to be pickled for multiprocessing) #####


# in order to create memory network, you need to modify this function
def load_file(full_name, debug=False):
	load_file_description = '''
	Inside load_file function\n
	load a *._conll file\n
	Input: full_name: path to the file\n
	Output: list of tuples for each conll doc in the file, where the tuple contains:\n
		(utts_text ([str]): list of the utterances in the document \n
		 utts_tokens ([[str]]): list of the tokens (conll words) in the document\n 
		 utts_corefs: list of coref objects (dicts) with the following properties:\n
			coref['label']: id of the coreference cluster,\n
			coref['start']: start index (index of first token in the utterance),\n
			coref['end': end index (index of last token in the utterance).\n
		 utts_speakers ([str]): list of the speaker associated to each utterances in the document\n 
		 name (str): name of the document\n
		 part (str): part of the document\n
		)
	'''
	#file_description = input(load_file_description)
	docs = []
	with io.open(full_name, 'rt', encoding='utf-8', errors='strict') as f:
		#lines = list(f)#.readlines()
		raw_lines = list(f)
		#lines = raw_lines[:145]
		lines = raw_lines
		utts_text = []
		utts_tokens = []
		utts_corefs = []
		utts_speakers = []
		tokens = []
		corefs = []
		#memnet_stories = [] # these stories will be appended to the main training data
		#memnet_sentence = [] # each individual setence of individual story
		index = 0
		speaker = ""
		name = ""
		part = ""

		#line_print = input(" Printing the lines and cols in the files read\n")
		for li, line in enumerate(lines):
			cols = line.split()
			if debug:
				print("line", li, "cols:", cols)
			# End of utterance
			# if end of collumn, then add the sentence
			if len(cols) == 0:
				if tokens:
					if debug:
						print("End of utterance")
					utts_text.append(u''.join(t + u' ' for t in tokens))
					utts_tokens.append(tokens)
					utts_speakers.append(speaker)
					utts_corefs.append(corefs)
					#memnet_story.append(memnet_sentence)
					tokens = []
					corefs = []
					index = 0
					speaker = ""
					continue
			# End of doc
			# At end of doc, add all the story till the point, and set story to new
			elif len(cols) == 2:
				if debug:
					print("End of doc")
					#doc1 = input()
				if cols[0] == "#end":
					if debug:
						print("Saving doc")
						#saving_doc1 = input()
					docs.append((utts_text, utts_tokens, utts_corefs, utts_speakers, name, part))
					utts_text = []
					utts_tokens = []
					utts_corefs = []
					utts_speakers = []
				else:
					raise ValueError("Error on end line " + line)
			# New doc
			elif len(cols) == 5:
				if debug:
					print("New doc")
					#new_doc1 = input()
				if cols[0] == "#begin":
					name = re.match(r"\((.*)\);", cols[2]).group(1)
					try:
						part = cols[4]
					except ValueError:
						print("Error parsing document part " + line)
					if debug:
						print("New doc", name, part, name[:2])
						#new_doc2 = input()
					tokens = []
					corefs = []
					index = 0
				else:
					raise ValueError("Error on begin line " + line)
			# Inside utterance
			elif len(cols) > 7:
				if debug:
					print("Inside utterance")
					#insie_utt = input()
				assert (cols[0] == name and int(cols[1]) == int(part)), "Doc name or part error " + line
				assert (int(cols[2]) == index), "Index error on " + line
				if speaker:
					assert (cols[9] == speaker), "Speaker changed in " + line + speaker
				else:
					speaker = cols[9]
					if debug:
						print("speaker", speaker)
						#speaker1 = input()
				if cols[-1] != u'-':
					coref_expr = cols[-1].split(u'|')
					if debug:
						print("coref_expr", coref_expr)
						#coref_expr_debug_1 = input()
					if not coref_expr:
						raise ValueError("Coref expression empty " + line)
					for tok in coref_expr:
						if debug:
							print("coref tok", tok)
							#coref_tok = input()
						try:
							match = re.match(r"^(\(?)(\d+)(\)?)$", tok)
						except:
							print("error getting coreferences for line " + line)
						assert match is not None, "Error parsing coref " + tok + " in " + line
						num = match.group(2)
						assert (num is not u''), "Error parsing coref " + tok + " in " + line
						if match.group(1) == u'(':
							if debug:
								print("New coref", num)
								#new_coref = input()
							corefs.append({'label': num, 'start': index, 'end': None})
						if match.group(3) == u')':
							j = None
							for i in range(len(corefs)-1, -1, -1):
								if debug:
									print("i", i)
									#ith_coref = input()
								if corefs[i]['label'] == num and corefs[i]['end'] is None:
									j = i
									break
							assert (j is not None), "coref closing error " + line
							if debug:
								print("End coref", num)
								#coref_end = input()
							corefs[j]['end'] = index
				# this is the part where you add the token, you need to add this token to story as well
				tokens.append(clean_token(cols[3]))
				index += 1
			else:
				raise ValueError("Line not standard " + line)
	#print("docs gathered is :")
	#uty = input()
	#print(docs[0])
	#hty = input()
	return docs

def set_feats(doc):
	doc.set_mentions_features()

def get_feats(doc, i):
	return doc.get_feature_array(doc_id=i)

def gather_feats(gathering_array, array, feat_name, pairs_ant_index, pairs_start_index):
	if gathering_array is None:
		gathering_array = array
	else:
		if feat_name == FEATURES_NAMES[6]:
			array = [a + pairs_ant_index for a in array]
		elif feat_name == FEATURES_NAMES[3]:
			array = [a + pairs_start_index for a in array]
		gathering_array += array
	return feat_name, gathering_array

def read_file(full_name):
	doc = ""
	with io.open(full_name, 'rt', encoding='utf-8', errors='strict') as f:
		doc = f.read()
	return doc

###################
### ConllDoc #####

class ConllDoc(Document):
	def __init__(self, name, part, *args, **kwargs):
		self.name = name
		self.part = part
		self.feature_matrix = {}
		self.conll_tokens = []
		self.conll_lookup = []
		self.gold_corefs = []
		self.missed_gold = []
		super(ConllDoc, self).__init__(*args, **kwargs)

	def get_conll_spacy_lookup(self, conll_tokens, spacy_tokens, debug=False):

		get_conll_spacy_lookup_description = '''
		Compute a look up table between spacy tokens (from spacy tokenizer)
		and conll pre-tokenized tokens
		Output: list[conll_index] => list of associated spacy tokens (assume spacy tokenizer has a finer granularity)
		'''
		#get_conll_spacy_lookup_input = input(get_conll_spacy_lookup_description)
		lookup = []
		c_iter = (t for t in conll_tokens)
		s_iter = enumerate(t for t in spacy_tokens)
		i, s_tok = next(s_iter)
		for c_tok in c_iter:
			if debug:
				print("conll", c_tok, "spacy", s_tok, "index", i)
			c_lookup = []
			while i is not None and len(c_tok) and c_tok.startswith(s_tok.text):
				c_lookup.append(i)
				c_tok = c_tok[len(s_tok):]
				i, s_tok = next(s_iter, (None, None))
				if debug and len(c_tok): print("eating token: conll", c_tok, "spacy", s_tok, "index", i)
			assert len(c_lookup), "Unmatched conll and spacy tokens"
			lookup.append(c_lookup)
		return lookup

	def add_conll_utterance(self, parsed, tokens, corefs, speaker_id, use_gold_mentions=False, debug=False):
		conll_lookup = self.get_conll_spacy_lookup(tokens, parsed)
		self.conll_tokens.append(tokens)
		self.conll_lookup.append(conll_lookup)
		# Convert conll tokens coref index in spacy tokens indexes
		#add_conll_utterance_input = input("Convert conll tokens coref index in spacy tokens indexes")
		identified_gold = [False] * len(corefs)
		for coref in corefs:
			assert (coref['label'] is not None and coref['start'] is not None and coref['end'] is not None), \
				("Error in coreference " + coref + " in " + parsed)
			coref['start'] = conll_lookup[coref['start']][0]
			coref['end'] = conll_lookup[coref['end']][-1]

		if speaker_id not in self.speakers:
			speaker_name = speaker_id.split(u'_')
			if debug:
				print("New speaker: ", speaker_id, "name: ", speaker_name)
				#new_speaker1 = input()
			self.speakers[speaker_id] = Speaker(speaker_id, speaker_name)
		if use_gold_mentions:
			for coref in corefs:
				#            print("coref['label']", coref['label'])
				#            print("coref text",parsed[coref['start']:coref['end']])
				mention = Mention(parsed[coref['start']:coref['end']], len(self.mentions), len(self.utterances),
								  self.n_sents, speaker=self.speakers[speaker_id], gold_label=coref['label'])
				self.mentions.append(mention)
				#            print("mention: ", mention, "label", mention.gold_label)
		else:
			#print("Parsed is : ",type(parsed))
			#print("len of utterances :",type(len(self.utterances)))
			#print("number of sentences :",type(self.n_sents))
			#print("speaker_id :",type(self.speakers[speaker_id]))
			#self._extract_mentions(parsed, len(self.utterances), self.n_sents, self.speakers[speaker_id])

			# find spans, create list of mentions, append them to self.mentions
			list_of_spans = extract_mentions_spans(parsed, blacklist=True, debug=False)
			for span in list_of_spans :
				mention = Mention(span, len(self.mentions), len(self.utterances),
								  self.n_sents, speaker=self.speakers[speaker_id], gold_label=None)
				self.mentions.append(mention)
			# Assign a gold label to mentions which have one
			if debug:
				print("Check corefs", corefs)
				#check_corefs_inp = input()
			#print(self.utterances)
			for i, coref in enumerate(corefs):
				for m in self.mentions:
					if m.utterance_index != len(self.utterances):
						continue
					if debug:
						print("Checking mention", m, m.utterance_index, m.start, m.end)
						#checking_mention_inp = input()
					if coref['start'] == m.start and coref['end'] == m.end - 1:
						m.gold_label = coref['label']
						identified_gold[i] = True
						if debug:
							print("Gold mention found:", m, coref['label'])
							#gold_mention_found = input()
			for found, coref in zip(identified_gold, corefs):
				if not found:
					self.missed_gold.append([self.name, self.part, str(len(self.utterances)), parsed.text, parsed[coref['start']:coref['end']+1].text])
					if debug:
						print("❄️ gold mention not in predicted mentions", coref, parsed[coref['start']:coref['end']+1])
						#gold_mention_not_found = input()
		self.utterances.append(parsed)
		self.gold_corefs.append(corefs)
		self.utterances_speaker.append(self.speakers[speaker_id])
		self.n_sents += len(list(parsed.sents))

	def get_single_mention_features_conll(self, mention, compressed=True):
		''' Compressed or not single mention features'''
		if not compressed:
			_, features = self.get_single_mention_features(mention)
			return features[np.newaxis, :]
		feat_l = [mention.features_["01_MentionType"],
				  mention.features_["02_MentionLength"],
				  mention.index,
				  len(self.mentions),
				  mention.features_["04_IsMentionNested"],
				  self.genre_,
				 ]
		return feat_l

	def get_pair_mentions_features_conll(self, m1, m2, compressed=True):
		''' Compressed or not single mention features'''
		if not compressed:
			_, features = self.get_pair_mentions_features(m1, m2)
			return features[np.newaxis, :]
		features_, _ = self.get_pair_mentions_features(m1, m2)
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

	def get_feature_array(self, doc_id, feature=None, compressed=True, debug=False):
		get_feature_array_description = """
		Prepare feature array:
			mentions_spans: (N, S)
			mentions_words: (N, W)
			mentions_features: (N, Fs)
			mentions_labels: (N, 1)
			mentions_pairs_start_index: (N, 1) index of beggining of pair list in pair_labels
			mentions_pairs_length: (N, 1) number of pairs (i.e. nb of antecedents) for each mention 
			pairs_features: (P, Fp)
			pairs_labels: (P, 1)
			pairs_ant_idx: (P, 1) => indexes of antecedents mention for each pair (mention index in doc)
		""" 
		#get_feature_array_input = input(get_feature_array_description)
		if not self.mentions:
			print("No mention in this doc !")
			return {}
		if debug:
			print("🛎 features matrices")
			#feature_matrices_input = input()
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
		#pairs_stories = []
		n_mentions = 0
		total_pairs = 0
		# need to create story from this mess 
		#if debug:
		#	print("mentions", self.mentions, str([m.gold_label for m in self.mentions]))
		#	mentions_matrice = input()
		for mention_idx, antecedents_idx in list(self.get_candidate_pairs(max_distance=None, max_distance_with_match=None)):
			n_mentions += 1
			mention = self.mentions[mention_idx]
			#print("PRINTING THE MENTION,",mention.text)
			#print("TYPE OF MENTION IS,",type(mention))
			'''print("MENTION UTTERANCE INDEX,",mention.utterance_index)
			print("MENTION START INDEX,",mention.start)
			print("MENTION END INDEX,",mention.end)
			print("UTTERANCES IN THIS CONTEXT,",self.utterances)
			print("TYPE OF UTTERANCES,",type(self.utterances))
			print("UTTERANCE REFERED,",self.utterances[mention.utterance_index])
			print("TYPE OF UTTERANCE,",type(self.utterances[mention.utterance_index]))
			print("MENTION EMBEDDING,",self.embed_extractor.get_word_embedding(mention))'''

			# let's create the story, story_embeds is the individual story, mentions_stories is the list of stories
			story_embeds = []
			for utt_index in range(mention.utterance_index) :
				for token in self.utterances[utt_index] :
					'''print("TOKEN ITERED,",token)
					print("TYPE OF TOKEN,",type(token))
					print("TOKEN EMBEDDING IS,",self.embed_extractor.get_word_embedding(token))'''
					#token_word, token_embed = self.embed_extractor.get_word_embedding(token)
					#print("TYPE OF EMBED_TOKEN,",type(token_embed))
					#story_embeds.append(token_embed.tolist())

					# since mention_words_idx works on Mention, we convert every token into a mention
					token_word_idx = word_idx_finder(self.embed_extractor,token.text)
					#story_embeds.append(token_embed.tolist())
					story_embeds.append(token_word_idx)
					#mentions_stories.append()
			for token_index in range(mention.start) :
				'''print("TOKEN IN FINAL LINE,",self.utterances[mention.utterance_index][token_index])
				print("TYPE OF TOKEN,",type(self.utterances[mention.utterance_index][token_index]))
				print("TOKEN EMBEDDING IN FINAL LINE,",self.embed_extractor.get_word_embedding(self.utterances[mention.utterance_index][token_index]))'''
				# the line below stores the pre defined word embeddings, we need the word_idx
				#token_word, token_embed = self.embed_extractor.get_word_embedding(self.utterances[mention.utterance_index][token_index])
				#token_word_idx = mention_words_idx(self.embed_extractor,self.utterances[mention.utterance_index][token_index])
				token_word_idx = word_idx_finder(self.embed_extractor,self.utterances[mention.utterance_index][token_index])
				#story_embeds.append(token_embed.tolist())
				story_embeds.append(token_word_idx)
			#print("STORY IS,",story_embeds)
			#print("TYPE OF STORY IS,",type(story_embeds))
			#print("STORY CREATED IS")
			#jy = input()
			#print(story_embeds)
			#shu = input()
			mentions_stories.append(story_embeds)

			#we_wait = input()
			mentions_spans.append(mention.spans_embeddings)
			w_idx = mention_words_idx(self.embed_extractor, mention)
			if w_idx is None:
				print("error in", self.name, self.part, mention.utterance_index)
			mentions_words.append(w_idx)
			mentions_features.append(self.get_single_mention_features_conll(mention, compressed))
			mentions_location.append([mention.start, mention.end, mention.utterance_index, mention_idx, doc_id])
			ants = [self.mentions[ant_idx] for ant_idx in antecedents_idx]
			no_antecedent = not any(ant.gold_label == mention.gold_label for ant in ants) or mention.gold_label is None
			if antecedents_idx:
				pairs_ant_idx += [idx for idx in antecedents_idx]
				pairs_features += [self.get_pair_mentions_features_conll(ant, mention, compressed) for ant in ants]
				ant_labels = [0 for ant in ants] if no_antecedent else [1 if ant.gold_label == mention.gold_label else 0 for ant in ants]
				pairs_labels += ant_labels
				#for idx in antecedents_idx :
				#	pairs_stories.append(story_embeds) # the same story will be appended multiple times depending upon the number of antecedents
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
					FEATURES_NAMES[6]: pairs_ant_idx if pairs_ant_idx else None,
					FEATURES_NAMES[7]: pairs_features if pairs_features else None,
					FEATURES_NAMES[8]: pairs_labels if pairs_labels else None,
					FEATURES_NAMES[9] : mentions_stories,
					FEATURES_NAMES[10]: [mentions_location],
					FEATURES_NAMES[11]: [self.conll_tokens],
					FEATURES_NAMES[12]: [self.conll_lookup],
					FEATURES_NAMES[13]: [{'name': self.name,
										  'part': self.part,
										  'utterances': list(str(u) for u in self.utterances),
										  'mentions': list(str(m) for m in self.mentions)}],
					}
		if debug:
			print("🚘 Summary")
			#summary_input = input()
			for k, v in out_dict.items():
				print(k, len(v))
		del mentions_features, mentions_labels, mentions_pairs_length, mentions_pairs_start, mentions_spans, mentions_words, pairs_ant_idx, pairs_features, pairs_labels, mentions_location, mentions_stories
		return n_mentions, total_pairs, out_dict

###################
### ConllCorpus #####
class ConllCorpus(object):
	def __init__(self, n_jobs=4, embed_path=PACKAGE_DIRECTORY+"/weights/", use_gold_mentions=False):
		self.n_jobs = n_jobs
		self.features = {}
		self.utts_text = []
		self.utts_tokens = []
		self.utts_corefs = []
		self.utts_speakers = []
		self.utts_doc_idx = []
		self.docs_names = []
		self.docs = []
		if embed_path is not None:
			self.embed_extractor = EmbeddingExtractor(embed_path)
		self.trainable_embed = []
		self.trainable_voc = []
		self.use_gold_mentions = use_gold_mentions

	def check_words_in_embeddings_voc(self, embedding, tuned=True, debug=False):
		print("🌋 Checking if words are in embedding voc")
		#check_words_in_embeddings_voc = input()
		if tuned:
			embed_voc = embedding.tun_idx
		else:
			embed_voc = embedding.stat_idx
		missing_words = []
		missing_words_sents = []
		missing_words_doc = []
		for doc in self.docs:
			if debug:
				print("Checking doc", doc.name, doc.part)
				#checking_doc_input = input()
			for sent in doc.utterances:
				if debug:
					print(sent.text)
					#sent_text_input = input()
				for word in sent:
					w = embedding.normalize_word(word)
					if debug:
						print(w)
						#word_input = input()
					if w not in embed_voc:
						missing_words.append(w)
						missing_words_sents.append(sent.text)
						missing_words_doc.append(doc.name + doc.part)
						if debug:
							out_str = "No matching tokens in tuned voc for " + w + \
									  " in sentence " + sent.text + \
									  " in doc " + doc.name + doc.part
							print(out_str)
							#out_str_input = input()
		return missing_words, missing_words_sents, missing_words_doc

	def test_sentences_words(self, save_file, debug=False):
		print("🌋 Saving sentence list")
		#test_sentences_list_input = input()
		with io.open(save_file, "w", encoding='utf-8') as f:
			if debug:
				print("Sentences saved in", save_file)
				#save_file = input()
			for doc in self.docs:
				out_str = "#begin document (" + doc.name + \
						  "); part " + doc.part + "\n"
				f.write(out_str)
				for sent in doc.utterances:
					f.write(sent.text + '\n')
				out_str = "#end document\n\n"
				f.write(out_str)

	def save_sentences(self, save_file, debug=False):
		print("🌋 Saving sentence list")
		with io.open(save_file, "w", encoding='utf-8') as f:
			if debug:
				print("Sentences saved in", save_file)
				#save_sentences_input = input()
			for doc in self.docs:
				out_str = "#begin document (" + doc.name + \
						  "); part " + doc.part + "\n"
				f.write(out_str)
				for sent in doc.utterances:
					f.write(sent.text + '\n')
				out_str = "#end document\n\n"
				f.write(out_str)

	def build_key_file(self, data_path, key_file, debug=False):
		print("🌋 Building key file from corpus")
		print("Saving in", key_file)
		#build_key_file_input = input()
		# Create a pool of processes. By default, one is created for each CPU in your machine.
		with io.open(key_file, "w", encoding='utf-8') as kf:
			if debug:
				print("Key file saved in", key_file)
				#key_file_saved = input()
			for dirpath, _, filenames in os.walk(data_path):
				print("In", dirpath)
				file_list = [os.path.join(dirpath, f) for f in filenames if f.endswith(".v4_auto_conll") \
							or f.endswith(".v4_gold_conll")]
				cleaned_file_list = []
				for f in file_list:
					fn = f.split('.')
					if fn[1] == "v4_auto_conll":
						gold = fn[0] + "." + "v4_gold_conll"
						if gold not in file_list:
							cleaned_file_list.append(f)
					else:
						cleaned_file_list.append(f)
			#self.load_file(file_list[0])
				#doc_list = parallel_process(cleaned_file_list, read_file)
				for file in cleaned_file_list:
					kf.write(read_file(file))

	def list_undetected_mentions(self, data_path, save_file, debug=False):
		self.read_corpus(data_path)
		print("🌋 Listing undetected mentions")
		#list_undetected_mentions = input()
		with io.open(save_file, 'w', encoding='utf-8') as out_file:
			for doc in tqdm(self.docs):
				for name, part, utt_i, utt, coref in doc.missed_gold:
					out_str = name + u"\t" + part + u"\t" + utt_i + u'\t"' + utt + u'"\n'
					out_str += coref + u"\n"
					out_file.write(out_str)
					if debug:
						print(out_str)
						#out_str_input = input()

	def read_corpus(self, data_path, debug=False):
		# this function holds the key to constructing the memory module for the conll corpus
		# find the discourse end marker, that holds the key to forming stories for memory inference
		print("🌋 Reading files")
		#read_corpus_input = input()
		dir_walk_count = 0
		for dirpath, _, filenames in os.walk(data_path):
			#dir_walk_count += 1
			#if dir_walk_count > 5 :
			#    break
			print("In", dirpath, os.path.abspath(dirpath))
			file_list = [os.path.join(dirpath, f) for f in filenames if f.endswith(".v4_auto_conll") \
						or f.endswith(".v4_gold_conll")]
			cleaned_file_list = []
			for f in file_list:
				fn = f.split('.')
				if fn[1] == "v4_auto_conll":
					gold = fn[0] + "." + "v4_gold_conll"
					if gold not in file_list:
						cleaned_file_list.append(f)
				else:
					cleaned_file_list.append(f)
			#doc_list = parallel_process(cleaned_file_list, load_file)

			# what is the doc list ?
			for file in cleaned_file_list:#executor.map(self.load_file, cleaned_file_list):
				docs = load_file(file)
				for utts_text, utt_tokens, utts_corefs, utts_speakers, name, part in docs:
					print("Imported", name)
					if debug:
						print("utts_text", utts_text)
						print("utt_tokens", utt_tokens)
						print("utts_corefs", utts_corefs)
						print("utts_speakers", utts_speakers)
						print("name, part", name, part)
						#utterances_in_doc_input = input()
					self.utts_text += utts_text
					self.utts_tokens += utt_tokens
					self.utts_corefs += utts_corefs
					self.utts_speakers += utts_speakers
					self.utts_doc_idx += [len(self.docs_names)] * len(utts_text)
					self.docs_names.append((name, part))
		print("utts_text size", len(self.utts_text))
		print("utts_tokens size", len(self.utts_tokens))
		print("utts_corefs size", len(self.utts_corefs))
		print("utts_speakers size", len(self.utts_speakers))
		print("utts_doc_idx size", len(self.utts_doc_idx))
		print("🌋 Building docs")
		for name, part in self.docs_names:
			self.docs.append(ConllDoc(name=name, part=part, nlp=None,
									  blacklist=False, consider_speakers=True,
									  embedding_extractor=self.embed_extractor,
									  conll=CONLL_GENRES[name[:2]]))
		print("🌋 Loading spacy model")
		try:
			spacy.info('en_core_web_sm')
			model = 'en_core_web_sm'
		except IOError:
			print("No spacy 2 model detected, using spacy1 'en' model")
			spacy.info('en')
			model = 'en'
		nlp = spacy.load(model)
		print("🌋 Parsing utterances and filling docs")
		doc_iter = (s for s in self.utts_text)
		for utt_tuple in tqdm(zip(nlp.pipe(doc_iter),
										   self.utts_tokens, self.utts_corefs,
										   self.utts_speakers, self.utts_doc_idx)):
			spacy_tokens, conll_tokens, corefs, speaker, doc_id = utt_tuple
			if debug:
				print(unicode_(self.docs_names[doc_id]), "-", spacy_tokens)
				#spacy_tokens_in_doc = input()
			doc = spacy_tokens
			if debug: 
				out_str = "utterance " + unicode_(doc) + " corefs " + unicode_(corefs) + \
						  " speaker " + unicode_(speaker) + "doc_id" + unicode_(doc_id)
				
				print(out_str.encode('utf-8'))
				print("CONLL TOKENS HERE ARE")
				print(Doc(nlp.vocab,conll_tokens))
				print("SENTENCE EMBEDDING OF TOKENS ARE")
				#print(self.embed_extractor.get_average_embedding(conll_tokens))
				#out_str_input2 = input()
			self.docs[doc_id].add_conll_utterance(doc, conll_tokens, corefs, speaker,
												  use_gold_mentions=self.use_gold_mentions)
			del spacy_tokens, conll_tokens, corefs,speaker, doc_id
		del nlp, doc_iter

	def build_and_gather_multiple_arrays(self, save_path,train_phase):
		print("🌋 Extracting mentions features")
		parallel_process(self.docs, set_feats, n_jobs=self.n_jobs)
		#for doc in self.docs :
		#	set_feats(doc)

		print("🌋 Building and gathering arrays")
		arr =[{'doc': doc,
			   'i': i} for i, doc in enumerate(self.docs)]
		#build_gather_array = input("Printing Array ")
		#print(arr)
		#arrays_dicts = parallel_process(arr, get_feats, use_kwargs=True, n_jobs=self.n_jobs)
		arrays_dicts = list()
		for arr_doc in arr :
			arrays_dicts.append(get_feats(arr_doc['doc'],arr_doc['i']))
		del arr
		gathering_dict = dict((feat, None) for feat in FEATURES_NAMES)
		n_mentions_list = []
		pairs_ant_index = 0
		pairs_start_index = 0
		for n, p, arrays_dict in tqdm(arrays_dicts):
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

		for feature in FEATURES_NAMES[:10]:
			print("Building numpy array for", feature, "length", len(gathering_dict[feature]))
			if feature != "mentions_spans":
				#array = np.array(gathering_dict[feature])
				# check if we are dealing with length of memories
				if feature == "mentions_stories" or feature == "pairs_stories" : 
					train_config = dict()
					max_story_len = 0
					if train_phase :
						max_story_len = max([len(story) for story in gathering_dict[feature]])
						max_story_len = min(200,max_story_len) # max length of the story is 30

						print("max story len, (in train phase should be 200)",max_story_len)
						if os.path.exists('train_config.pickle'):
							file_handle_init = open('train_config.pickle','rb')
							train_config = pickle.load(file_handle_init)
							file_handle_init.close()
						file_handle = open('train_config.pickle','wb')
						train_config[feature] = max_story_len
						pickle.dump(train_config,file_handle)
						file_handle.close()
					else :
						file_handle = open('train_config.pickle','rb')
						train_config = pickle.load(file_handle)
						max_story_len = train_config[feature]
						print("max story len is (should be 200),",max_story_len)
						file_handle.close()

					#append_list = [0] # 1 is the embedding size, because now the story cosists of word_idx
					#append_list = 50*[0] # 50 is the embedding size
					#print(type(append_list))
					gathering_array = []
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
			np.save(save_path + feature, array)
		for feature in FEATURES_NAMES[9:]:
			print("Saving pickle", feature, "size", len(gathering_dict[feature]))
			with open(save_path + feature + '.bin', "wb") as fp:  
				pickle.dump(gathering_dict[feature], fp)
		del arrays_dicts, gathering_dict

	def save_vocabulary(self, save_path, debug=False):
		def _vocabulary_to_file(path, vocabulary):
			print("🌋 Saving vocabulary")
			with io.open(path, "w", encoding='utf-8') as f:
				if debug:
					print('voc saved in {path}, length: {size}'
						  .format(path=path, size=len(vocabulary)))
				for w in tunable_voc:
					f.write(w + '\n')

		print("🌋 Building tunable vocabulary matrix from static vocabulary")
		tunable_voc = self.embed_extractor.tun_voc
		_vocabulary_to_file(
			path=save_path + 'tuned_word_vocabulary.txt',
			vocabulary=tunable_voc
		)

		static_voc = self.embed_extractor.stat_voc
		_vocabulary_to_file(
			path=save_path + 'static_word_vocabulary.txt',
			vocabulary=static_voc
		)

		tuned_word_embeddings = np.vstack([self.embed_extractor.get_stat_word(w)[1] for w in tunable_voc])
		print("Saving tunable voc, size:", tuned_word_embeddings.shape)
		np.save(save_path + "tuned_word_embeddings", tuned_word_embeddings)

		static_word_embeddings = np.vstack([self.embed_extractor.static_embeddings[w] for w in static_voc])
		print("Saving static voc, size:", static_word_embeddings.shape)
		np.save(save_path + "static_word_embeddings", static_word_embeddings)


if __name__ == '__main__':
	DIR_PATH = os.path.dirname(os.path.realpath(__file__))
	parser = argparse.ArgumentParser(description='Training the neural coreference model')
	parser.add_argument('--function', type=str, default='all', help='Function ("all", "key", "parse", "find_undetected")')
	parser.add_argument('--path', type=str, default=DIR_PATH + '/data/', help='Path to the dataset')
	parser.add_argument('--key', type=str, help='Path to an optional key file for scoring')
	parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs (default 1)')
	parser.add_argument('--train_phase',dest='train_phase',default=False,action='store_true', help='Is it the training phase')
	args = parser.parse_args()
	if args.key is None:
		args.key = args.path + "/key.txt"
	CORPUS = ConllCorpus(n_jobs=args.n_jobs)
	if args.function == 'parse' or args.function == 'all':
		SAVE_DIR = args.path + "/numpy/"
		if not os.path.exists(SAVE_DIR):
			os.makedirs(SAVE_DIR)
		else:
			if os.listdir(SAVE_DIR):
				print("There are already data in", SAVE_DIR)
				print("Erasing")
				for file in os.listdir(SAVE_DIR):
					print(file)
					os.remove(SAVE_DIR + file)
		start_time = time.time()
		CORPUS.read_corpus(args.path)
		print('=> read_corpus time elapsed', time.time() - start_time)
		start_time2 = time.time()
		train_phase_key = args.train_phase
		#print("train_phase is ",train_phase_key)
		#nhd = input()
		CORPUS.build_and_gather_multiple_arrays(SAVE_DIR,args.train_phase)
		print('=> build_and_gather_multiple_arrays time elapsed', time.time() - start_time2)
		start_time2 = time.time()
		CORPUS.save_vocabulary(SAVE_DIR)
		print('=> save_vocabulary time elapsed', time.time() - start_time2)
		print('=> total time elapsed', time.time() - start_time)
	if args.function == 'key' or args.function == 'all':
		CORPUS.build_key_file(args.path, args.key)
	if args.function == 'find_undetected':
		CORPUS.list_undetected_mentions(args.path, args.path + "/undetected_mentions.txt")

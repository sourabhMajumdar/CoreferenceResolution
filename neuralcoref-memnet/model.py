# coding: utf8
"""Conll training algorithm"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

#from utils import to_var
import copy
import math

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in, dropout=0.5):
        super(Model, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.pair_top = nn.Sequential(nn.Linear(D_pair_in + embedding_dim, H1), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H3, 1),
                                      nn.Linear(1, 1))
        self.single_top = nn.Sequential(nn.Linear(D_single_in + embedding_dim, H1), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H3, 1),
                                        nn.Linear(1, 1))
        self.init_weights()

        self.hops = 3
        self.embd_size = embedding_dim
        self.temporal_encoding = False
        self.position_encoding = False

        init_rng = 0.1
        # memnn added below
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for _ in range(self.hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0, init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0] # query encoder
        self.vocab_size = vocab_size

        # Temporal Encoding: see 4.1
        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))
            self.TC = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))

    def init_weights(self):
        w = (param.data for name, param in self.named_parameters() if 'weight' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform(self.word_embeds.weight.data, a=-0.5, b=0.5)
        for t in w:
            nn.init.xavier_uniform(t)
        for t in b:
            nn.init.constant(t, 0)

    def load_embeddings(self, preloaded_weights):
        self.word_embeds.weight = nn.Parameter(preloaded_weights)

    def load_weights(self, weights_path):
        print("Loading weights")
        single_layers_weights, single_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("single_mention_weights"):
                single_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("single_mention_bias"):
                single_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_single_linear = (layer for layer in self.single_top if isinstance(layer, nn.Linear))
        for w, b, layer in zip(single_layers_weights, single_layers_biases, top_single_linear):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())
        pair_layers_weights, pair_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("pair_mentions_weights"):
                pair_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("pair_mentions_bias"):
                pair_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_pair_linear = (layer for layer in self.pair_top if isinstance(layer, nn.Linear))
        for w, b, layer in zip(pair_layers_weights, pair_layers_biases, top_pair_linear):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())

    def forward(self, inputs, concat_axis=1):
        pairs = (len(inputs) == 9)
        if pairs:
            #print("Running Pairwise model")
            spans, words, single_features, ant_spans, ant_words, ana_spans, ana_words, pair_features, pairs_story = inputs
            #print(spans.size())
            #print(words.size())
            #print(single_features.size())
            #print(ant_spans.size())
            #print(ant_words.size())
            #print(ana_spans.size())
            #print(ana_words.size())
            #print(pair_features.size())
            #print(pairs_story.size())
            #hy = input()
            x = pairs_story
            bs = x.size(0) # bs stands for batch size
            story_len = x.size(1) # story len stands for the length of the story
            #s_sent_len = pairs_story.size(2) # sent len stands for the length of single sentence 
        else:
            #print("Running Single Mention Model")
            spans, words, single_features, mentions_story = inputs
            #print(spans.size())
            #print(words.size())
            #print(single_features.size())
            #print(mentions_story.size())
            #he = input()
            x = mentions_story
            bs = mentions_story.size(0) # bs stands for batch size
            story_len = mentions_story.size(1) # story len stands for the length of the story
            #s_sent_len = mentions_story.size(2) # sent len stands for the length of single sentence 

        

        # Position Encoding
        if self.position_encoding:
            J = s_sent_len
            d = self.embd_size
            pe = to_var(torch.zeros(J, d)) # (s_sent_len, embd_size)
            for j in range(1, J+1):
                for k in range(1, d+1):
                    l_kj = (1 - j / J) - (k / d) * (1 - 2 * j / J)
                    pe[j-1][k-1] = l_kj
            pe = pe.unsqueeze(0).unsqueeze(0) # (1, 1, s_sent_len, embd_size)
            pe = pe.repeat(bs, story_len, 1, 1) # (bs, story_len, s_sent_len, embd_size)

        #x = x.view(bs,1,story_len,-1)
        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)
        #print(x.size())
        #x = x.view(bs,story_len,-1)
        #print(x.size())
        #print("END OF EMBEDD EXP")
        #print("SHAPE OF WORDS,",words.size())
        #print("SHAPE OF EMBED WORDS,",self.embd_size)
        q = words

        u = self.dropout(self.B(q)) # (bs, q_sent_len, embd_size)
        print(u.size())
        u = torch.sum(u, 1) # (bs, embd_size)

        # Adjacent weight tying
        for k in range(self.hops):
            #print(x.size())
            #print(type(x))
            #x = x.view(bs*story_len,-1)
            #print(x.size())
            #print("IN THE SHAPE OF SENT")
            #print(x)
            #print(self.vocab_size)
            #print(words)
            #print(words.size())
            #ghu = input("PRTNTING THE SHAPE OF TENSOR")
            m = self.A[k](x)
            #m = self.dropout(self.A[k](x))            # (bs*story_len, s_sent_len, embd_size)
            #m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            m = m.view(bs,story_len,-1) # (bs,story_len,embd_size)
            if self.position_encoding:
                m *= pe # (bs, story_len, s_sent_len, embd_size)
            #m = torch.sum(m, 2) # (bs, story_len, embd_size)
            if self.temporal_encoding:
                m += self.TA.repeat(bs, 1, 1)[:, :story_len, :]

            c = self.dropout(self.A[k+1](x))           # (bs*story_len, s_sent_len, embd_size)
            #c = c.view(bs, story_len, s_sent_len, -1)  # (bs, story_len, s_sent_len, embd_size)
            c = c.view(bs,story_len,-1)
            #c = torch.sum(c, 2)                        # (bs, story_len, embd_size)
            if self.temporal_encoding:
                c += self.TC.repeat(bs, 1, 1)[:, :story_len, :] # (bs, story_len, embd_size)

            #p = torch.bmm(m, u.unsqueeze(2)).squeeze() # (bs, story_len)
            #u = u.view(bs,1,-1)
            #print("SHAPE OF C,",c.size())
            #print("SHAPE OF M,",m.size())
            #print("SHAPE OF U,",u.size())
           # hf = input()
            #u_unsq = u.unsqueeze(2)
            #print(u_unsq.size())
            #mkmv = input("unsqueezed Tensor")
            p = torch.bmm(m, u.unsqueeze(2)).squeeze() # (bs, story_len)
            if bs == 1 :
                p = p.view(1,-1)
            print("initial shape of p,",p.size())
            #mcw = input("printed shape of p")
            p = F.softmax(p, -1).unsqueeze(1)          # (bs, 1, story_len)
            #print("SHAPE OF P,",p.size())
            o = torch.bmm(p, c).squeeze(1)             # use m as c, (bs, embd_size)
            u = o + u # (bs, embd_size)

        # Don't need this because we are not interested in attention over candidates rather the embedding size only
        #W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        #out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)

        # instead of returning the log_softmax, we need to return the score
        #return F.log_softmax(out, -1)
        embed_words = self.drop(self.word_embeds(words).view(words.size()[0], -1))
        #print("NORMAL WORDS ARE EMBEDDED")
        #print(spans.size())
        #print(embed_words.size())
        #print(single_features.size())
        #print(out.size())
        single_input = torch.cat([spans, embed_words, single_features,u], 1)
        single_scores = self.single_top(single_input)
        if pairs:
            batchsize, pairs_num, _ = ana_spans.size()
            u = u.view(bs,1,-1)
            #print("SHAPE BEFORE REPEATING,",u.size())
            u = u.repeat(1,pairs_num,1)
            ant_embed_words = self.drop(self.word_embeds(ant_words.view(batchsize, -1)).view(batchsize, pairs_num, -1))
            ana_embed_words = self.drop(self.word_embeds(ana_words.view(batchsize, -1)).view(batchsize, pairs_num, -1))
            #print("INSIDE THE PAIR INPUT FUNCTION")
            #print(ant_spans.size())
            #print(ant_embed_words.size())
            #print(ana_spans.size())
            #print(ana_embed_words.size())
            #print(pair_features.size())
            #print(out.size())
            pair_input = torch.cat([ant_spans, ant_embed_words, ana_spans, ana_embed_words, pair_features,u], 2)
            pair_scores = self.pair_top(pair_input).squeeze(dim=2)
            total_scores = torch.cat([pair_scores, single_scores], concat_axis)
        return total_scores if pairs else single_scores



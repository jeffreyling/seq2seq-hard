#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import pdb
import re
import pickle
from collections import defaultdict

def build_embeds(fname, outfile, words):
    def load_bin_vec(fname, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        return word_vecs

    word_vecs = load_bin_vec(fname, words)
    embeds = np.random.uniform(-0.25, 0.25, (len(words), len(word_vecs.values()[0])))
    embeds[0] = 0 # padding
    for word, vec in word_vecs.iteritems():
        embeds[words[word]-1] = vec

    f = h5py.File(outfile, "w")
    f["word_vecs"] = np.array(embeds)
    f.close()

class Indexer:
    def __init__(self, symbols = ["*blank*","<unk>","<d>","</d>"]):
        self.vocab = defaultdict(int) # special dictionary type for counting
        self.PAD = symbols[0]
        self.UNK = symbols[1]
        self.BOS = symbols[2]
        self.EOS = symbols[3]
        self.d = {self.PAD: 1, self.UNK: 2, self.BOS: 3, self.EOS: 4}

    def add_w(self, ws):
        for w in ws:
            if w not in self.d:
                self.d[w] = len(self.d) + 1
            
    def convert(self, w):
        return self.d[w] if w in self.d else self.d[self.UNK]

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def clean(self, s):
        s = s.replace(self.PAD, "")
        s = s.replace(self.BOS, "")
        s = s.replace(self.EOS, "")
        return s
        
    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

    def prune_vocab(self, k):
        vocab_list = [(word, count) for word, count in self.vocab.iteritems()]
        vocab_list.sort(key = lambda x: x[1], reverse=True)
        k = min(k, len(vocab_list))
        self.pruned_vocab = {pair[0]:pair[1] for pair in vocab_list[:k]}
        for word in self.pruned_vocab:
            if word not in self.d:
                self.d[word] = len(self.d) + 1

    def load_vocab(self, vocab_file):
        self.d = {}
        for line in open(vocab_file, 'r'):
            v, k = line.strip().split()
            self.d[v] = int(k)

def pad(ls, length, symbol, no_cut=False):
    if len(ls) >= length:
        if no_cut:
          return ls
        else:
          return ls[:length]
    return ls + [symbol] * (length -len(ls))

def pad_front(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]
    return [symbol] * (length -len(ls)) + ls

def get_data(args):
    src_indexer = Indexer(["<blank>","<unk>","<d>","</d>"]) # dummy
    word_indexer = Indexer(["<blank>","<unk>","<s>", "</s>"]) # for both source and target
    word_indexer.add_w([src_indexer.BOS, src_indexer.EOS])

    def make_vocab(srcfile, targetfile, seqlength, max_sent_l=0, truncate=0, no_pad=0, targetseqlength=100, pretrain=0):
        num_docs = 0
        num_blank_docs = 0
        num_bad_targs = 0
        for i, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())

            src = src_orig.strip().strip("</s>").split("</s>") # splits the doc into sentences
            targ = targ_orig.strip().split()
            if len(src) < 1 or len(src[0]) < 1:
              num_blank_docs += 1
              continue
            if len(src) > seqlength and no_pad == 0:
              if truncate == 1:
                src = src[:seqlength]
              else:
                continue
            if pretrain == 0:
              if len(targ) < 1 or len(targ) > targetseqlength:
                num_bad_targs += 1
                continue

            if pretrain > 0:
                num_docs += pretrain
            else:
                num_docs += 1

            for word in targ:
                word_indexer.vocab[word] += 1

            for sent in src:
                sent = word_indexer.clean(sent)
                if len(sent) == 0:
                    continue
                words = sent.split() 
                max_sent_l = max(len(words)+2, max_sent_l)
                for word in words:
                    word_indexer.vocab[word] += 1                                        

            # print src, targ
            # raw_input()
                
        return max_sent_l, num_docs, num_blank_docs, num_bad_targs
                
    def convert(srcfile, targetfile, batchsize, seqlength, outfile, num_docs,
                max_sent_l, max_doc_l=0, unkfilter=0, shuffle=0,
                truncate=0, no_pad=0, repeat_words=0, targetseqlength=100, pretrain=0):
        
        if no_pad == 1:
            newseqlength = seqlength
        else:
            newseqlength = seqlength + 2 #add 2 for EOS and BOS; length in sents of the longest document
        newtargetseqlength = targetseqlength + 2 #add 2 for EOS and BOS

        targets = np.zeros((num_docs, newtargetseqlength), dtype=int) # the target sequence
        target_output = np.zeros((num_docs, newtargetseqlength), dtype=int) # next word to predict
        sources = np.zeros((num_docs, newseqlength), dtype=int) # input split into sentences
        source_lengths = np.zeros((num_docs,), dtype=int) # lengths of each document
        target_lengths = np.zeros((num_docs,), dtype=int) # lengths of each target sequence
        sources_word = np.zeros((num_docs, newseqlength, max_sent_l), dtype=int) # input by word
        source_word_l = np.zeros((num_docs,), dtype=int) # max sentence length for doc
        dropped = 0
        doc_id = 0
        for _, (src_orig, targ_orig) in \
                enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
            src_orig = src_indexer.clean(src_orig.strip())
            targ_orig = word_indexer.clean(targ_orig.strip())
            orig_sents = src_orig.strip().strip("</s>").split("</s>")
            if no_pad == 1:
                src = orig_sents
            else:
                src = [src_indexer.BOS] + orig_sents + [src_indexer.EOS]
            if pretrain > 0:
                targ = []
                for j in xrange(pretrain):
                    sampled_sents = np.random.choice(orig_sents, min(2, len(orig_sents)), replace=False)
                    sampled_sents = [' '.join(sent.strip().split()[:targetseqlength/2]) for sent in list(sampled_sents)]
                    sampled_sents = ' '.join(sampled_sents).strip().split()
                    targ.append([word_indexer.BOS] + sampled_sents[:targetseqlength] + [word_indexer.EOS])
            else:
                targ = [word_indexer.BOS] + targ_orig.strip().split() + [word_indexer.EOS]
            max_doc_l = max(len(src), max_doc_l)

            # drop
            if no_pad == 1:
              if len(src) < 1 or len(src[0]) < 1:
                dropped += 1
                continue                   
            else:
              if len(src) < 3 or len(src[1]) < 3:
                dropped += 1
                continue
            if len(src) > newseqlength and no_pad == 0:
              if truncate == 1:
                src = src[:newseqlength]
              else:
                dropped += 1
                continue                   
            if pretrain == 0:
                if len(targ) > newtargetseqlength or len(targ) < 3:
                    dropped += 1
                    continue

            if pretrain > 0:
                for j in xrange(len(targ)):
                    targ[j] = pad(targ[j], newtargetseqlength+1, word_indexer.PAD)
                    for word in targ[j]:
                        #use UNK for target, but not for source
                        word = word if word in word_indexer.d else word_indexer.UNK
                    targ[j] = word_indexer.convert_sequence(targ[j])
                    targ[j] = np.array(targ[j], dtype=int)
            else:
                targ = pad(targ, newtargetseqlength+1, word_indexer.PAD)
                for word in targ:
                    #use UNK for target, but not for source
                    word = word if word in word_indexer.d else word_indexer.UNK
                targ = word_indexer.convert_sequence(targ)
                targ = np.array(targ, dtype=int)

            if no_pad == 1:
              src = pad(src, newseqlength, src_indexer.PAD, no_cut=True)
            else:
              src = pad(src, newseqlength, src_indexer.PAD)
            src_word = []
            for sent in src:
                sent = word_indexer.clean(sent)
                if no_pad == 1:
                  if sent == '': continue
                  word = sent.split() + [word_indexer.EOS]
                  src_word = src_word + word
                else:
                  word = [word_indexer.BOS] + sent.split() + [word_indexer.EOS]
                  if len(word) > max_sent_l:
                      word = word[:max_sent_l]
                      word[-1] = word_indexer.EOS
                  word_idx = word_indexer.convert_sequence(pad(word, max_sent_l, word_indexer.PAD))
                  src_word.append(word_idx)
            if no_pad == 1:
                src = src[:newseqlength]
            src = [1 if x == src_indexer.PAD else 0 for x in src] # 1 if pad, 0 o.w.
            src = np.array(src, dtype=int) # not useful
            
            if unkfilter > 0:
                targ_unks = float((targ[:-1] == 2).sum())
                src_unks = float((src == 2).sum())                
                if unkfilter < 1: #unkfilter is a percentage if < 1
                    targ_unks = targ_unks/(len(targ[:-1])-2)
                    src_unks = src_unks/(len(src)-2)
                if targ_unks > unkfilter or src_unks > unkfilter:
                    dropped += 1
                    continue
                
            if pretrain > 0:
                for j in xrange(len(targ)):
                  targets[doc_id] = np.array(targ[j][:-1],dtype=int) # get all but the last pad
                  target_lengths[doc_id] = (targets[doc_id] != 1).sum()
                  target_output[doc_id] = np.array(targ[j][1:],dtype=int)                    
                  sources[doc_id] = np.array(src, dtype=int)
                  source_lengths[doc_id] = (sources[doc_id] != 1).sum()            
                  if no_pad == 1:
                    src_word = word_indexer.convert_sequence(pad(src_word, newseqlength*max_sent_l, word_indexer.PAD))
                    result = np.array(src_word, dtype=int).reshape((newseqlength, max_sent_l))
                    if repeat_words > 0:
                      idx = 0
                      for j in xrange(newseqlength):
                        result[j] = src_word[idx:idx+max_sent_l]
                        idx = idx + max_sent_l - repeat_words # subtract deficit
                    sources_word[doc_id] = result
                  else:
                    sources_word[doc_id] = np.array(src_word, dtype=int)
                  source_word_l[doc_id] = (sources_word[doc_id] != 1).sum(1).max() # get max sent len for doc

                  doc_id += 1
            else:
                targets[doc_id] = np.array(targ[:-1],dtype=int) # get all but the last pad
                target_lengths[doc_id] = (targets[doc_id] != 1).sum()
                target_output[doc_id] = np.array(targ[1:],dtype=int)                    
                sources[doc_id] = np.array(src, dtype=int)
                source_lengths[doc_id] = (sources[doc_id] != 1).sum()            
                if no_pad == 1:
                  src_word = word_indexer.convert_sequence(pad(src_word, newseqlength*max_sent_l, word_indexer.PAD))
                  result = np.array(src_word, dtype=int).reshape((newseqlength, max_sent_l))
                  if repeat_words > 0:
                    idx = 0
                    for j in xrange(newseqlength):
                      result[j] = src_word[idx:idx+max_sent_l]
                      idx = idx + max_sent_l - repeat_words # subtract deficit
                  sources_word[doc_id] = result
                else:
                  sources_word[doc_id] = np.array(src_word, dtype=int)
                source_word_l[doc_id] = (sources_word[doc_id] != 1).sum(1).max() # get max sent len for doc

                doc_id += 1
            if doc_id % 100000 == 0:
                print("{}/{} docs processed".format(doc_id, num_docs))

        print(doc_id, num_docs)
        if shuffle == 1:
            rand_idx = np.random.permutation(doc_id)
            targets = targets[rand_idx]
            target_output = target_output[rand_idx]
            sources = sources[rand_idx]
            source_lengths = source_lengths[rand_idx]
            target_lengths = target_lengths[rand_idx]
            sources_word = sources_word[rand_idx]
            source_word_l = source_word_l[rand_idx]
        
        #break up batches based on source lengths
        # get source_lengths into a particular shape then sort by length
        source_lengths = source_lengths[:doc_id]
        source_sort = np.argsort(source_lengths) 

        sources = sources[source_sort]
        targets = targets[source_sort]
        target_output = target_output[source_sort]
        target_l = target_lengths[source_sort] # define new arrays to be the lengths
        source_l = source_lengths[source_sort]

        curr_l = 0
        l_location = [] #idx where sent length changes
        
        for j,i in enumerate(source_sort): # iterate over the indices of the sorted sentences
            if source_lengths[i] > curr_l:
                curr_l = source_lengths[i]
                l_location.append(j+1)
        l_location.append(len(sources)) # l_location is array where sentence length changes happen

        #get batch sizes
        curr_idx = 1
        batch_idx = [1]
        nonzeros = [] # number of non padding entries
        batch_l = [] # batch lengths (number of docs in the batch)
        batch_w = [] # batch widths (length of the docs in the batch)
        target_l_max = []
        for i in range(len(l_location)-1): # iterate over all the different document lengths
            while curr_idx < l_location[i+1]:
                curr_idx = min(curr_idx + batchsize, l_location[i+1])
                batch_idx.append(curr_idx)
        for i in range(len(batch_idx)-1): # iterate over batch_idx
            batch_l.append(batch_idx[i+1] - batch_idx[i])            
            batch_w.append(source_l[batch_idx[i]-1])
            nonzeros.append((target_output[batch_idx[i]-1:batch_idx[i+1]-1] != 1).sum().sum())
            target_l_max.append(max(target_l[batch_idx[i]-1:batch_idx[i+1]-1]))

        # Write output
        f = h5py.File(outfile, "w")

        # NOTE: not changing the names of things so don't need to change data.lua
        f["source"] = sources # sources is now binary where 0 = not pad, 1 = pad, not a useful matrix
        f["target"] = targets
        f["target_output"] = target_output
        f["target_l"] = np.array(target_l_max, dtype=int)
        f["target_l_all"] = target_l        
        f["batch_l"] = np.array(batch_l, dtype=int)
        f["batch_w"] = np.array(batch_w, dtype=int) # source_l
        f["batch_idx"] = np.array(batch_idx[:-1], dtype=int)
        f["target_nonzeros"] = np.array(nonzeros, dtype=int)
        f["source_size"] = np.array([2]) #np.array([len(src_indexer.d)]) we don't care about this
        f["target_size"] = np.array([len(word_indexer.d)])
        f["char_size"] = np.array([len(word_indexer.d)])
        del sources, targets, target_output
        sources_word = sources_word[source_sort]
        source_word_l = source_word_l[source_sort]
        f["source_char"] = sources_word
        f["source_char_l"] = np.array(source_word_l, dtype=int)
        del sources_word, source_word_l
        print("Saved {} documents (dropped {} due to length/unk filter)".format(
            len(f["source"]), dropped))
        f.close()                
        return max_doc_l

    print("First pass through data to get vocab...")
    max_sent_l, num_docs_train, num_blank_docs, num_bad_targs = make_vocab(args.srcfile, args.targetfile,
                                             args.seqlength, 0, args.truncate, args.no_pad, args.targetseqlength, args.pretrain)
    print("Number of docs in training: {}".format(num_docs_train))
    print("Blank docs: {}, bad targs: {}".format(num_blank_docs, num_bad_targs))
    max_sent_l, num_docs_valid, num_blank_docs, num_bad_targs = make_vocab(args.srcvalfile, args.targetvalfile,
                                             args.seqlength, max_sent_l, args.truncate, args.no_pad, args.targetseqlength, args.pretrain)
    print("Number of docs in valid: {}".format(num_docs_valid))    
    print("Blank docs: {}, bad targs: {}".format(num_blank_docs, num_bad_targs))
    print("Max sentence length (before cutting): {}".format(max_sent_l))
    max_sent_l = min(max_sent_l, args.maxsentlength)
    print("Max sentence length (after cutting): {}".format(max_sent_l))
    print("Target sequence length: {}".format(args.targetseqlength))

    #prune and write vocab
    word_indexer.prune_vocab(args.srcvocabsize)
    if args.charvocabfile != '':
        print('Loading pre-specified char vocab from ' + args.charvocabfile)
        word_indexer.load_vocab(args.charvocabfile)

    word_indexer.write(args.outputfile + ".word.dict")
    print("Word vocab size: Original = {}, Pruned = {}".format(len(word_indexer.vocab), 
                                                          len(word_indexer.d)))

    if args.word2vec != '':
        print('Building embeddings from ' + args.word2vec)
        build_embeds(args.word2vec, args.outputfile + "-word2vec.hdf5", word_indexer.d)

    max_doc_l = 0
    max_doc_l = convert(args.srcvalfile, args.targetvalfile, args.batchsize, args.seqlength,
                         args.outputfile + "-val.hdf5", num_docs_valid,
                         max_sent_l, max_doc_l, args.unkfilter, args.shuffle,
                         args.truncate, args.no_pad, args.repeat_words, args.targetseqlength, args.pretrain)
    max_doc_l = convert(args.srcfile, args.targetfile, args.batchsize, args.seqlength,
                         args.outputfile + "-train.hdf5", num_docs_train, max_sent_l,
                         max_doc_l, args.unkfilter, args.shuffle,
                         args.truncate, args.no_pad, args.repeat_words, args.targetseqlength, args.pretrain)
    
    print("Max doc length (before dropping): {}".format(max_doc_l))
    
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srcvocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                " Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--targetvocabsize', help="Size of target vocabulary, constructed "
                                                "by taking the top X most frequent words. "
                                                "Rest are replaced with special UNK tokens.",
                                                type=int, default=50000)
    parser.add_argument('--charvocabfile', help="If working with a preset vocab, "
                                         "then including this use the char vocab provided here.",
                                          type = str, default='')    
    parser.add_argument('--srcfile', help="Path to source training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--targetfile', help="Path to target training data, "
                                           "where each line represents a single "
                                           "source/target sequence.", required=True)
    parser.add_argument('--srcvalfile', help="Path to source validation data.", required=True)
    parser.add_argument('--targetvalfile', help="Path to target txt validation data.", required=True)
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=64)
    parser.add_argument('--seqlength', help="Maximum sequence (document) length. Sequences longer "
                                               "than this are dropped.", type=int, default=20)
    parser.add_argument('--targetseqlength', help="Maximum sequence (document) length. Sequences longer "
                                               "than this are dropped.", type=int, default=100)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ", type=str, required=True)
    parser.add_argument('--maxsentlength', help="For the character models, words are "
                                           "(if longer than maxwordlength) or zero-padded "
                                            "(if shorter) to maxwordlength", type=int, default=30)
    parser.add_argument('--word2vec', help="Path to word2vec", type = str, default='')
    parser.add_argument('--unkfilter', help="Ignore sentences with too many UNK tokens. "
                                       "Can be an absolute count limit (if > 1) "
                                       "or a proportional limit (0 < unkfilter < 1).",
                                          type = float, default = 0)
    parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on  "
                                           "source length).",
                                          type = int, default = 0)
    parser.add_argument('--truncate', help="If = 1, truncate docs instead of dropping.",
                                          type = int, default = 0)
    parser.add_argument('--no_pad', help="If = 1, fill image instead of padding sentences",
                                          type = int, default = 0)
    parser.add_argument('--repeat_words', help="If > 1, repeat words at end of each row to next row",
                                          type = int, default = 0)
    parser.add_argument('--pretrain', help="If > 1, emit pretraining data - target is number of random sents from source",
                                          type = int, default = 0)
    parser.add_argument('--sample_sents', help="If = 1, sample sentences from doc up to seqlength",
                                          type = int, default = 0)
    parser.add_argument('--seed', help="random seed", type = int, default = 3435)

    args = parser.parse_args(arguments)
    np.random.seed(args.seed)
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

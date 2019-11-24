# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-09-11
#
# Distributed under terms of the MIT license.


import torch
import os
import pdb

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

def read_subjs_data(filename):
  subjs_file = open(filename)
  subjs = []
  for line in subjs_file:
    subjs.append(torch.LongTensor([int(idx) for idx in line.strip().split()]))
  return subjs


def read_tags(tag_file_path):
  tags = []
  with open(tag_file_path) as f:
    for line in f:
      tags.append([int(line.strip())])
  return torch.Tensor(tags)


def read_subjs_data_set(filename):
  subjs_file = open(filename)
  subjs = []
  attrs = []
  for line in subjs_file:
    fields = line.strip().split(" ||| ")
    subjs_in_line = torch.IntTensor(eval(fields[0]))
    attrs_in_line = torch.IntTensor(eval(fields[1]))
    if attrs_in_line.size(0) == 0:
      pdb.set_trace()
    subjs.append(subjs_in_line)
    attrs.append(attrs_in_line)

  return subjs, attrs


class SentCorpus(object):
  def __init__(self, path, dictionary, append_eos=True):
    self.dictionary = dictionary
    self.test = self.tokenize(path, append_eos)

  def tokenize(self, path, append_eos=True):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Tokenize file content
    ids = []
    unk_idx = self.dictionary.word2idx["<unk>"]
    with open(path, 'r') as f:
        for line in f:
            if append_eos:
                words = line.split() + ["<eos>"]
            else:
                words = line.split()
            sent = torch.LongTensor(len(words))
            for idx, word in enumerate(words):
                sent[idx] = self.dictionary.word2idx.get(word, unk_idx)
            ids.append(sent)

    return ids




# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-05-01
#
# Distributed under terms of the MIT license.

import argparse
import data
import logging
import pdb
import torch
import torch.nn.functional as F

from utils import batchify2
from salience import SalienceManager
from salience import SalienceType

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="compute salience from awd language model")
opt_parser.add_argument("--save", type=str, metavar="PATH", required=True, help="path to the saved model")
opt_parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to test data (without the .prefx.txt, .verbs.txt, .subjs.txt suffix)")
opt_parser.add_argument("--dict-data", type=str, metavar="PATH", required=True, help="path to the data used to build dict")
opt_parser.add_argument("--output", type=str, metavar="PATH", required=True, help="path to output file")
opt_parser.add_argument("--batch-size", type=int, default=10, help="test batch size")
opt_parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")


def model_load(fn):
  with open(fn, 'rb') as f:
    # model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
    model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer


def get_normalized_probs(output, weight, log_probs=True):
  logits = torch.matmul(output, weight.transpose(0, 1))  # (samples, vocab_size)
  if log_probs:
    return F.log_softmax(logits, dim=-1)
  else:
    return F.softmax(logits, dim=-1)


def compute_salience(prefix_data, verbs_data, subjs_data, model, batch_size=10, cuda=False):
  for (prefix_batch, prefix_mask), (verbs_batch, _), (subjs_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(verbs_data[0], verbs_data[1]), zip(subjs_data[0], subjs_data[1])):
    padded_seq_len = prefix_batch.size(0)
    hidden = model.init_hidden(batch_size)
    if cuda:
      prefix_batch = prefix_batch.cuda()  # (src_len, bsz * n_samples)  # TODO may need some expanding
      verbs_batch = verbs_batch.cuda()
      prefix_mask = prefix_mask.cuda()
      subjs_batch = subjs_batch.cuda()  # (2, bsz)
    output, hidden = model(prefix_batch, hidden)
    probs = get_normalized_probs(output, model.decoder.weight).view(padded_seq_len, batch_size, -1)

    final_prefix_index = torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1  # (1, bsz)
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    final_word_probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)

    SalienceManager.backward_with_salience_single_timestep(final_word_probs, verbs_batch[0, :], model)
    averaged_salience_verum = SalienceManager.average_single_timestep()  # (src_len, bsz)
    SalienceManager.clear_salience()
    SalienceManager.backward_with_salience_single_timestep(final_word_probs, verbs_batch[1, :], model)
    averaged_salience_malum = SalienceManager.average_single_timestep()  # (src_len, bsz)
    SalienceManager.clear_salience()

    verum_subj_salience = torch.gather(averaged_salience_verum, 0, subjs_batch[0, :].unsqueeze(0))  # (1, bsz)
    malum_subj_salience = torch.gather(averaged_salience_malum, 0, subjs_batch[1, :].unsqueeze(0))  # (1, bsz)

    # TODO calculate score conditioned on verb verum/malum probability


def main(options):
  import os
  import hashlib
  fn = 'corpus.{}.data'.format(hashlib.md5(options.dict_data.encode()).hexdigest())
  if os.path.exists(fn):
    print('Loading cached dataset...')
    trn_corpus = torch.load(fn)
  else:
    print('Producing dataset...')
    trn_corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

  model, _, _ = model_load(options.save)
  model.train()
  model.encoder.activate(SalienceType.vanilla)
  prefix_corpus = data.SentCorpus(options.data_prefix + ".prefx.txt", trn_corpus.dictionary)
  verbs_corpus = data.SentCorpus(options.data_prefix + ".verbs.txt", trn_corpus.dictionary)
  subjs_corpus = data.read_subjs_data(options.data_prefix + ".subjs.txt")
  prefix_data = batchify2(prefix_corpus.test, options.batch_size, prefix_corpus.dictionary.word2idx["<eos>"])
  verbs_data = batchify2(verbs_corpus.test, options.batch_size, prefix_corpus.dictionary.word2idx["<eos>"])
  subjs_data = batchify2(subjs_corpus, options.batch_size, -1)  # there won't be pad for this
  compute_salience(prefix_data, verbs_data, subjs_data, model, options.batch_size, options.cuda)


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)

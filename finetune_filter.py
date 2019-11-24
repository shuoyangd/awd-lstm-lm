# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-07-03
#
# Distributed under terms of the MIT license.

import argparse
import data
import logging
import torch
import torch.nn.functional as F
# cudnn backward cannot be called at eval mode
import torch.backends.cudnn as cudnn
cudnn.enabled = False

from utils import batchify2, batchify3
from salience import SalienceManager
from salience import SalienceType

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="fine-tune a filter for salience test on awd language model")
opt_parser.add_argument("--model", type=str, metavar="PATH", required=True, help="path to the saved model")
opt_parser.add_argument("--outdir", type=str, metavar="PATH", required=True, help="path to the finetuned model")
opt_parser.add_argument("--data-prefix", type=str, metavar="PATH", required=True, help="path to training and dev data (without the .prefx.txt, tag.txt suffix)")
opt_parser.add_argument("--dict-data", type=str, metavar="PATH", required=True, help="path to the data used to build dict")
opt_parser.add_argument("--batch-size", type=int, default=10, help="test batch size")
opt_parser.add_argument("--cuda", action='store_true', default=False, help="use cuda")

opt_parser.add_argument("--max-epoch", type=int, default=10)
opt_parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam"], default="SGD")
opt_parser.add_argument("--momentum", type=float, default=0.99, help="momentum for SGD")
opt_parser.add_argument("--learning-rate", '-lr', type=float, default=0.1)


def read_tags(tag_file_path):
  tags = []
  with open(tag_file_path) as f:
    for line in f:
      tags.append([int(line.strip())])
  return torch.Tensor(tags)


def model_load(fn):
  with open(fn, 'rb') as f:
    # model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
    model, criterion, optimizer = torch.load(f)
    return model, criterion, optimizer

def get_normalized_probs(output, weight, bias=None, log_probs=True):
  logits = torch.matmul(output, weight.transpose(0, 1))  # (samples, vocab_size)
  if bias is not None:
    logits += bias
  if log_probs:
    return F.log_softmax(logits, dim=-1)
  else:
    return F.softmax(logits, dim=-1)


def finetune(prefix_data, tag_data, model, outdir, optimizer, epoch_n, cuda=False):
  iter_ = 0
  for (prefix_batch, prefix_mask), (tag_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(tag_data[0], tag_data[1])):

    optimizer.zero_grad()
    batch_size = tag_batch.size(1)
    padded_seq_len = prefix_batch.size(0)
    hidden = model.init_hidden(batch_size)

    if cuda:
        prefix_batch = prefix_batch.cuda()
        prefix_mask = prefix_mask.cuda()
        tag_batch = tag_batch.cuda()

    model.reset()
    model.encoder.deactivate()
    output, _ = model(prefix_batch, hidden)
    probs = get_normalized_probs(output, model.decoder.weight, model.decoder.bias).view(padded_seq_len, batch_size, -1)  # (seq_len, bsz, vocab_size)

    final_prefix_index = (torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1).clamp_(0)  # (1, bsz) TODO: there is -1 index
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)

    loss = -torch.sum(torch.gather(probs, 1, tag_batch.transpose(0, 1)))  # (bsz, 1)
    # loss = -torch.sum(probs[:, tag_batch])
    loss.backward()
    optimizer.step()

    iter_ += 1
    if iter_ % 100 == 0:
      print("training loss at {0} is {1}".format(iter_, loss.item() / batch_size))

  torch.save(model, "{0}.epoch{1}".format(outdir, epoch_n))
  return model


def validate(prefix_data, tag_data, model, cuda=False):
  raw_loss = 0.0
  n_samples = 0
  for (prefix_batch, prefix_mask), (tag_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(tag_data[0], tag_data[1])):

    batch_size = tag_batch.size(1)
    padded_seq_len = prefix_batch.size(0)

    if padded_seq_len > 150:
        continue

    n_samples += batch_size
    hidden = model.init_hidden(batch_size)

    if cuda:
        prefix_batch = prefix_batch.cuda()
        prefix_mask = prefix_mask.cuda()
        tag_batch = tag_batch.cuda()

    model.reset()
    model.encoder.deactivate()
    output, _ = model(prefix_batch, hidden)
    probs = get_normalized_probs(output, model.decoder.weight, model.decoder.bias).view(padded_seq_len, batch_size, -1)  # (seq_len, bsz, vocab_size)

    final_prefix_index = (torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1).clamp_(0)  # (1, bsz) TODO: there is -1 index
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)

    loss = -torch.sum(torch.gather(probs, 1, tag_batch.transpose(0, 1)))  # (bsz, 1)
    raw_loss += loss

  return raw_loss / n_samples


def main(options):
  import os
  import hashlib
  fn = 'corpus.{}.data'.format(hashlib.md5(options.dict_data.encode()).hexdigest())
  if os.path.exists(fn):
    print('Loading cached dataset...')
    trn_corpus = torch.load(fn)
  else:
    print('Producing dataset...')
    trn_corpus = data.Corpus(options.dict_data)
    torch.save(trn_corpus, fn)

  model, _, _ = model_load(options.model)
  ntoken, nhid = model.decoder.weight.size()
  for param in model.parameters():
    param.requires_grad = False
  filter_ = torch.nn.Linear(nhid, 2)  # TODO: for other filters, it may be ntoken
  filter_.weight.data.uniform_(-0.1, 0.1)
  if options.cuda:
    filter_ = filter_.cuda()
  model.decoder = filter_
  assert model.decoder.weight.requires_grad == True

  if options.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.decoder.parameters(), lr=options.learning_rate, momentum=options.momentum)
  else:
    optimizer = eval("torch.optim." + options.optimizer)(model.decoder.parameters(), lr=options.learning_rate)

  prefix_corpus_train = data.SentCorpus(options.data_prefix + "train.prefx.txt", trn_corpus.dictionary)
  tag_corpus_train = read_tags(options.data_prefix + "train.tag.txt")
  prefix_corpus_dev = data.SentCorpus(options.data_prefix + "valid.prefx.txt", trn_corpus.dictionary)
  tag_corpus_dev = read_tags(options.data_prefix + "valid.tag.txt")

  # batchify3: pads at the beginning, may introduce slight bias, but should be fine overall
  prefix_data_train = batchify2(prefix_corpus_train.test, options.batch_size, trn_corpus.dictionary.word2idx["<eos>"])
  tag_data_train = batchify2(tag_corpus_train, options.batch_size, trn_corpus.dictionary.word2idx["<eos>"])
  prefix_data_dev = batchify2(prefix_corpus_dev.test, options.batch_size, trn_corpus.dictionary.word2idx["<eos>"])
  tag_data_dev = batchify2(tag_corpus_dev, options.batch_size, trn_corpus.dictionary.word2idx["<eos>"])

  for epoch in range(options.max_epoch):
    model = finetune(prefix_data_train, tag_data_train, model, options.outdir, optimizer, epoch, options.cuda)
    print("valid loss after epoch {0}: {1}".format(epoch, validate(prefix_data_dev, tag_data_dev, model, options.cuda)))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)


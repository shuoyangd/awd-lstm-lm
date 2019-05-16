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
# cudnn backward cannot be called at eval mode
import torch.backends.cudnn as cudnn
cudnn.enabled = False

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
opt_parser.add_argument("--salience-type", type=str, choices=["vanilla", "smoothed", "integral", "guided"], default="vanilla", help="type of salience")
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


def evaluate(prefix_data, verbs_data, subjs_data, model, outdir, cuda=False):
  total_count = 0
  match_count = 0
  pos_count = 0
  neg_count = 0
  pos_match_count = 0
  neg_match_count = 0
  verum_salience_output = open(outdir + ".verum", 'w')
  malum_salience_output = open(outdir + ".malum", 'w')
  # bsz and batch_size are the same thing :)
  for (prefix_batch, prefix_mask), (verbs_batch, _), (subjs_batch, _) in \
      zip(zip(prefix_data[0], prefix_data[1]), zip(verbs_data[0], verbs_data[1]), zip(subjs_data[0], subjs_data[1])):
    batch_size = subjs_batch.size(1)
    sample_size = 1
    if model.encoder.salience_type == SalienceType.smoothed:
      sample_size = model.encoder.smooth_samples
    elif model.encoder.salience_type == SalienceType.integral:
      sample_size = model.encoder.integral_steps
    bsz_samples = batch_size * sample_size

    padded_seq_len = prefix_batch.size(0)
    hidden = model.init_hidden(bsz_samples)
    no_salience_hidden = model.init_hidden(batch_size)
    if cuda:
      prefix_batch = prefix_batch.cuda()  # (src_len, bsz)  # TODO may need some expanding
      verbs_batch = verbs_batch.cuda()
      prefix_mask = prefix_mask.cuda()
      subjs_batch = subjs_batch.cuda()  # (2, bsz) XXX: shouldn't be expanded

    # first evaluation: get probability of two verbs
    real_salience_type = model.encoder.salience_type
    model.salience_type = None
    model.reset()
    model.encoder.deactivate()
    no_salience_output, _ = model(prefix_batch, no_salience_hidden)
    no_salience_probs = get_normalized_probs(no_salience_output, model.decoder.weight).view(padded_seq_len, batch_size, -1)

    # second evaluation: salience computation
    model.reset()
    model.encoder.activate(real_salience_type)
    output, hidden = model(prefix_batch, hidden)

    probs = get_normalized_probs(output, model.decoder.weight).view(padded_seq_len, bsz_samples, -1)

    final_prefix_index = torch.sum(prefix_mask, dim=0).unsqueeze(0) - 1  # (1, bsz)
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, probs.size(-1))  # (1, batch_size, vocab_size)
    no_salience_verb_probs = torch.gather(no_salience_probs, 0, final_prefix_index).squeeze(0)  # (bsz, vocab_size)
    # this is the probability conditioned on ALL the words in the prefix, i.e., the probability of the verb
    final_prefix_index = final_prefix_index.unsqueeze(2).expand(-1, -1, sample_size, -1).contiguous().view(1, bsz_samples, -1)  # (1, batch_size * sample_size, vocab_size)
    verb_probs = torch.gather(probs, 0, final_prefix_index).squeeze(0)  # (bsz * n_samples, vocab_size)

    SalienceManager.backward_with_salience_single_timestep(verb_probs, verbs_batch[0, :], model)
    averaged_salience_verum = SalienceManager.average_single_timestep()  # (src_len, bsz)
    SalienceManager.clear_salience()
    SalienceManager.backward_with_salience_single_timestep(verb_probs, verbs_batch[1, :], model)
    averaged_salience_malum = SalienceManager.average_single_timestep()  # (src_len, bsz)
    SalienceManager.clear_salience()

    verum_salience_output.write(str(averaged_salience_verum.squeeze().tolist()) + "\n")
    malum_salience_output.write(str(averaged_salience_malum.squeeze().tolist()) + "\n")

    pos_verum_subj_salience = torch.gather(averaged_salience_verum, 0, subjs_batch[0, :].unsqueeze(0)).squeeze()  # (bsz,)
    pos_malum_subj_salience = torch.gather(averaged_salience_malum, 0, subjs_batch[0, :].unsqueeze(0)).squeeze()  # (bsz,)
    neg_verum_subj_salience = torch.gather(averaged_salience_verum, 0, subjs_batch[1, :].unsqueeze(0)).squeeze()
    neg_malum_subj_salience = torch.gather(averaged_salience_malum, 0, subjs_batch[1, :].unsqueeze(0)).squeeze()

    # XXX: must use no salience verb probs here
    # salience methods that involve samples will change the verb probability here
    # causing small variance in the test applied
    verum_probs = torch.gather(no_salience_verb_probs, -1, verbs_batch[0, :].unsqueeze(1).expand(batch_size, sample_size)).view(batch_size, -1)  # (bsz, n_samples)
    malum_probs = torch.gather(no_salience_verb_probs, -1, verbs_batch[1, :].unsqueeze(1).expand(batch_size, sample_size)).view(batch_size, -1)  # (bsz, n_samples)
    verum_probs = torch.mean(verum_probs, dim=1)  # (bsz,)
    malum_probs = torch.mean(malum_probs, dim=1)  # (bsz,)

    pos_match = (pos_verum_subj_salience > pos_malum_subj_salience) * (verum_probs > malum_probs)
    neg_match = (neg_verum_subj_salience < neg_malum_subj_salience) * (verum_probs < malum_probs)
    match = pos_match + neg_match

    total_count += subjs_batch.size(1)
    match_count += torch.sum(match)
    pos_count += torch.sum((verum_probs > malum_probs))
    neg_count += torch.sum((verum_probs < malum_probs))
    pos_match_count += torch.sum(pos_match)
    neg_match_count += torch.sum(neg_match)
    logging.info("running frac: {0} / {1} = {2}".format(match_count, total_count, match_count.item() / total_count))
    if pos_count != 0:
        logging.info("running pos frac: {0} / {1} = {2}".format(pos_match_count, pos_count, pos_match_count.item() / pos_count.item()))
    else:
        logging.info("running pos frac: 0 / 0 = 0")
        pos_count = 1
    if neg_count != 0:
        logging.info("running neg frac: {0} / {1} = {2}".format(neg_match_count, neg_count, neg_match_count.item() / neg_count.item()))
    else:
        logging.info("running neg frac: 0 / 0 = 0")
        neg_count = 1

  verum_salience_output.close()
  malum_salience_output.close()
  return (match_count.item(), total_count, match_count.item() / total_count, \
          pos_match_count.item(), pos_count.item(), pos_match_count.item() / pos_count.item(), \
          neg_match_count.item(), neg_count.item(), neg_match_count.item() / neg_count.item())


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

  model, _, _ = model_load(options.save)
  model.eval()
  model.encoder.activate(eval("SalienceType." + options.salience_type))
  prefix_corpus = data.SentCorpus(options.data_prefix + ".prefx.txt", trn_corpus.dictionary)
  verbs_corpus = data.SentCorpus(options.data_prefix + ".verbs.txt", trn_corpus.dictionary)
  subjs_corpus = data.read_subjs_data(options.data_prefix + ".subjs.txt")
  prefix_data = batchify2(prefix_corpus.test, options.batch_size, prefix_corpus.dictionary.word2idx["<eos>"])
  verbs_data = batchify2(verbs_corpus.test, options.batch_size, prefix_corpus.dictionary.word2idx["<eos>"])
  subjs_data = batchify2(subjs_corpus, options.batch_size, -1)  # there won't be pad for this
  frac = evaluate(prefix_data, verbs_data, subjs_data, model, options.output, options.cuda)
  print(frac)


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)

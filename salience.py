# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-04-26
#
# Distributed under terms of the MIT license.
#
# Using this code snippet in your research work requires you to cite the following paper:
# @inproceedings{DBLP:conf/wmt/DingXK19,
#   author    = {Shuoyang Ding and
#                Hainan Xu and
#                Philipp Koehn},
#   title     = {Saliency-driven Word Alignment Interpretation for Neural Machine Translation},
#   booktitle = {Proceedings of the Fourth Conference on Machine Translation, {WMT}
#                2019, Florence, Italy, August 1-2, 2019 - Volume 1: Research Papers},
#   pages     = {1--12},
#   year      = {2019},
#   crossref  = {DBLP:conf/wmt/2019-1},
#   url       = {https://www.aclweb.org/anthology/W19-5201/},
#   timestamp = {Mon, 26 Aug 2019 14:06:00 +0200},
#   biburl    = {https://dblp.org/rec/bib/conf/wmt/DingXK19},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }
#

from enum import Enum
import pdb

import torch
import torch.nn as nn

class SalienceType(Enum):
  vanilla=1  # word salience in Ding et al. (2019)
  smoothed=2  # word salience with SmoothGrad in Ding et al. (2019)
  integral=3
  li=4
  li_smoothed=5


class SalienceManager:
  single_sentence_salience = []  # each element in this list corresponds to one target word of shape (bsz * samples, src_len)
  __bsz = 1
  __n_samples = None

  @classmethod
  def set_n_samples(cls, n_samples):
    cls.__n_samples = n_samples

  @classmethod
  def compute_salience(cls, grad):
    # grad = torch.clamp(grad, min=0.0).detach().cpu()
    cls.single_sentence_salience.append(grad)  # no need to skip words

  @classmethod
  def compute_li_et_al_saliency(cls, grad):
    grad = torch.mean(torch.abs(grad), dim=-1).detach()
    cls.single_sentence_salience.append(grad / torch.sum(grad, dim=0).unsqueeze(1))

  @classmethod
  def extend_salience(cls, grad):
    """
    This is used when both source and target salience score needs to be computed.
    """
    grad = torch.clamp(grad, min=0.0).detach()
    last_grad = cls.single_sentence_salience[-1]
    last_grad = torch.cat([grad, last_grad], dim=1)  # we do care about eos though, as it's a separate input token
    # cls.single_sentence_salience[-1] = last_grad
    cls.single_sentence_salience[-1] = last_grad / torch.sum(last_grad, dim=1).unsqueeze(1)

  @classmethod
  def backward_with_salience(cls, probs, target, model, sequential=False):
    """
    probs: (bsz * n_samples, target_len, vocab_size) output probability distribution with regard to a input sentence
    target: (bsz, target_len) target word to evaluate salience score on
    """
    cls.__bsz, tlen = target.size()
    if model.encoder.salience_type == SalienceType.smoothed and not sequential:
        cls.__n_samples = model.encoder.smooth_samples
    elif model.encoder.salience_type == SalienceType.integral and not sequential:
        cls.__n_samples = model.encoder.integral_steps
    else:
        cls.__n_samples = 1
    target_probs = torch.gather(probs, -1, target).view(cls.__bsz, cls.n_samples, tlen)  # (batch_size, n_samples, target_len)
    # this mean is taken mainly for speed reason
    # otherwise, we would have to iterate through n_samples as well, which is not necessary
    # as gradient will be 0 for prediction score that does not correspond to the input sample
    target_probs = torch.mean(target_probs, dim=1)  # (batch_size, target_len)
    for i in range(cls.__bsz):
      for j in range(tlen):
        target_probs[i, j].backward(retain_graph=True)
        model.zero_grad()

  @classmethod
  def backward_with_salience_single_timestep(cls, probs, target, model, sequential=False):
    """
    probs: (bsz * n_samples, vocab_size) output probability distribution of a single time step with regard to a input sentence
    target: (bsz,) target word corresponding to a single time step, to evaluate salience score on
    """
    cls.__bsz = target.size()[0]
    if model.encoder.salience_type == SalienceType.smoothed and not sequential:
        cls.__n_samples = model.encoder.smooth_samples
    elif model.encoder.salience_type == SalienceType.integral and not sequential:
        cls.__n_samples = model.encoder.integral_steps
    else:
        cls.__n_samples = 1
    target = target.unsqueeze(1).expand(cls.__bsz, cls.__n_samples).contiguous().view(-1)
    target_probs = torch.gather(probs, -1, target.unsqueeze(1)).view(cls.__bsz, cls.__n_samples)  # (batch_size, n_samples)
    target_probs = torch.mean(target_probs, dim=1)  # (batch_size,)
    for i in range(cls.__bsz):
      target_probs[i].backward(retain_graph=True)
      model.zero_grad()

  @classmethod
  def average(cls):
    stacked_salience = torch.stack(cls.single_sentence_salience, dim=1)  # (bsz * n_samples, tgt_len, src_len)
    bsz_samples, tgt_len, src_len = stacked_salience.size()
    stacked_salience = stacked_salience.view(cls.__bsz, cls.__n_samples, tgt_len, src_len)
    averaged_salience = torch.mean(stacked_salience, dim=1)
    return averaged_salience

  @classmethod
  def average_single_timestep(cls):
    # the second dimension is of size bsz because we did averaging before backprop
    stacked_salience = torch.stack(cls.single_sentence_salience, dim=1)  # (src_len, bsz, bsz * n_samples)
    stacked_salience = torch.sum(stacked_salience, dim=1)  # (src_len, bsz * n_samples)
    src_len, bsz_samples = stacked_salience.size()
    stacked_salience = stacked_salience.view(src_len, cls.__bsz, cls.__n_samples)
    averaged_salience = torch.mean(stacked_salience, dim=2)
    return averaged_salience

  @classmethod
  def clear_salience(cls):
    cls.single_sentence_salience = []

class SalienceEmbedding(nn.Embedding):

  def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
               max_norm=None, norm_type=2., scale_grad_by_freq=False,
               sparse=False, _weight=None,
               salience_type=None,
               smooth_factor=0.15, smooth_samples=30,  # for smoothgrad
               integral_steps=100):
    """
    Has all the usual functionality of the normal word embedding class in PyTorch
    but will also set the proper hooks to compute word-level salience score when
    salience_type is set and self.training != True.

    Should be used together with SalienceManager.
    """

    super(SalienceEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx,
                                            max_norm, norm_type, scale_grad_by_freq,
                                            sparse, _weight)
    self.salience_type = salience_type
    self.smooth_factor = smooth_factor
    self.smooth_samples = smooth_samples
    self.integral_steps = integral_steps
    self.activated = False


  def activate(self, salience_type, sequential=False):
    """
    Salience should not be computed for all evaluations, like validation during training.
    In these cases, SalienceEmbedding should act the same way as normal word embedding.
    This switch should be turned on when salience needs to be computed.
    """
    self.activated = True
    self.salience_type = salience_type
    self.sequential = sequential


  def set_sequential_alpha(self, alpha):
    if not self.sequential:
      raise Exception("cannot set sequential alpha when salience embedding is not sequential")
    else:
      self.sequential_alpha = alpha


  def deactivate(self):
    self.activated = False


  def forward(self, input):
    """
    Note that this module is slightly more constrained on the shape of input than
    the original nn.Embedding class.

    We assume that the second dimension is the batch size.
    """

    batch_size = input.size(1)
    if self.salience_type and self.activated:
      # in case where multiple samples are needed
      # repeat the samples and set accompanying parameters accordingly
      orig_size = list(input.size())
      new_size = orig_size
      new_size_expand = None
      if self.salience_type == SalienceType.smoothed and not self.sequential:
        new_size_expand = tuple([ orig_size[0], orig_size[1], self.smooth_samples ] + orig_size[2:])
      elif self.salience_type == SalienceType.integral and not self.sequential:
        new_size_expand = tuple([ orig_size[0], orig_size[1], self.integral_steps ] + orig_size[2:])

      if new_size_expand:
        new_size = tuple([new_size_expand[0], new_size_expand[1] * new_size_expand[2]] + list(new_size_expand[3:]))
        input = input.unsqueeze(2).expand(*new_size_expand).contiguous().view(*new_size)

    # normal embedding query
    x = super(SalienceEmbedding, self).forward(input)

    if self.salience_type and self.activated:
      sel = torch.ones_like(input).float()
      sel.requires_grad = True
      sel.register_hook(lambda grad: SalienceManager.compute_salience(grad))

      if not (self.salience_type == SalienceType.li or \
            self.salience_type == SalienceType.li_smoothed):
        xp = x.permute(2, 0, 1)
        if self.salience_type == SalienceType.integral and not self.sequential:
          alpha = torch.arange(0, 1, 1 / self.integral_steps) + 1 / self.integral_steps  # (0, 1] rather than [0, 1)
          alpha = alpha.unsqueeze(0).expand(batch_size, self.integral_steps)
          alpha = alpha.contiguous().view(batch_size * self.integral_steps, -1).squeeze()
          alpha = alpha.type_as(x)  # (batch_size * integral_steps)
          xp = xp * sel * alpha
        elif self.salience_type == SalienceType.integral:  # sequential IG
          xp = xp * sel * self.sequential_alpha
        else:
          xp = xp * sel

        x = xp.permute(1, 2, 0)
      else:
        x.register_hook(lambda grad: SalienceManager.compute_li_et_al_saliency(grad))

      if (self.salience_type == SalienceType.smoothed or \
            self.salience_type == SalienceType.li_smoothed) and \
            self.smooth_factor > 0.0:
          x = x + torch.normal(torch.zeros_like(x), \
                  torch.ones_like(x) * self.smooth_factor * (torch.max(x) - torch.min(x)))

    return x


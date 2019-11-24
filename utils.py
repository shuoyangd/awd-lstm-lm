import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def batchify2(data, batch_size, pad_index, order=None):
    """

    :param data: [(sent_len,)]
    :param batch_size:
    :param pad_index:
    :param order: (optional) desired order of data
    :return [(seq_len, batch_size)]
    """
    if order is not None:
      data = [data[i] for i in order]
    batchized_data = []
    batchized_mask = []
    # except for last batch
    for start_i in range(0, len(data) - batch_size, batch_size):
      batch_data = data[start_i: start_i + batch_size]
      seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
      batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
      mask_tensor = torch.zeros((seq_len, batch_size)).byte()
      for idx, sent_data in enumerate(batch_data):
        # batch_tensor[:, idx] = truncate_or_pad(sent_data, 0, seq_len, pad_index=pad_index)
        batch_tensor[0:len(sent_data), idx] = sent_data
        mask_tensor[0:len(sent_data), idx] = 1
      batchized_data.append(batch_tensor)
      batchized_mask.append(mask_tensor)

    # last batch
    if len(data) % batch_size != 0:
        batch_data = data[len(data) // batch_size * batch_size:]
    else:
        batch_data = data[(len(data) // batch_size - 1) * batch_size:]

    seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
    final_batch_size = len(batch_data)
    batch_tensor = (torch.ones((seq_len, final_batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, final_batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)

    return batchized_data, batchized_mask

def batchify3(data, batch_size, pad_index, order=None):
    """

    :param data: [(sent_len,)]
    :param batch_size:
    :param pad_index:
    :param order: (optional) desired order of data
    :return [(seq_len, batch_size)]
    """
    if order is not None:
      data = [data[i] for i in order]
    batchized_data = []
    batchized_mask = []
    # except for last batch
    for start_i in range(0, len(data) - batch_size, batch_size):
      batch_data = data[start_i: start_i + batch_size]
      seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
      batch_tensor = (torch.ones((seq_len, batch_size)) * pad_index).long()
      mask_tensor = torch.zeros((seq_len, batch_size)).byte()
      for idx, sent_data in enumerate(batch_data):
        # batch_tensor[:, idx] = truncate_or_pad(sent_data, 0, seq_len, pad_index=pad_index)
        batch_tensor[seq_len-len(sent_data):, idx] = sent_data
        mask_tensor[seq_len-len(sent_data):, idx] = 1
      batchized_data.append(batch_tensor)
      batchized_mask.append(mask_tensor)

    # last batch
    if len(data) % batch_size != 0:
        batch_data = data[len(data) // batch_size * batch_size:]
    else:
        batch_data = data[(len(data) // batch_size - 1) * batch_size:]

    seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
    final_batch_size = len(batch_data)
    batch_tensor = (torch.ones((seq_len, final_batch_size)) * pad_index).long()
    mask_tensor = torch.zeros((seq_len, final_batch_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      batch_tensor[seq_len-len(sent_data):, idx] = sent_data
      mask_tensor[seq_len-len(sent_data):, idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)

    return batchized_data, batchized_mask

def batchify4(data, batch_size, pad_index, order=None):
    """

    :param data: [(sent_len,)]
    :param batch_size:
    :param pad_index:
    :param order: (optional) desired order of data
    :return [(seq_len, batch_size)]
    """
    if order is not None:
      data = [data[i] for i in order]
    batchized_data = []
    batchized_mask = []
    item_size = list(data[0].size())  # the first dim is the seq_len, which will change for different items
    item_size.insert(1, batch_size)
    batched_item_size = item_size
    # except for last batch
    for start_i in range(0, len(data) - batch_size, batch_size):
      batch_data = data[start_i: start_i + batch_size]
      seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
      batched_item_size[0] = seq_len
      batch_tensor = (torch.ones(tuple(batched_item_size)) * pad_index).long()
      mask_tensor = torch.zeros(tuple(batched_item_size)).byte()
      for idx, sent_data in enumerate(batch_data):
        # batch_tensor[:, idx] = truncate_or_pad(sent_data, 0, seq_len, pad_index=pad_index)
        batch_tensor[0:len(sent_data), idx] = sent_data
        mask_tensor[0:len(sent_data), idx] = 1
      batchized_data.append(batch_tensor)
      batchized_mask.append(mask_tensor)

    # last batch
    if len(data) % batch_size != 0:
        batch_data = data[len(data) // batch_size * batch_size:]
    else:
        batch_data = data[(len(data) // batch_size - 1) * batch_size:]

    seq_len = max([len(batch_data[i]) for i in range(len(batch_data))])  # find longest seq
    final_batch_size = len(batch_data)
    batched_item_size[0] = seq_len
    batched_item_size[1] = final_batch_size
    batch_tensor = (torch.ones(tuple(batched_item_size)) * pad_index).long()
    mask_tensor = torch.zeros(tuple(batched_item_size)).byte()
    for idx, sent_data in enumerate(batch_data):
      batch_tensor[0:len(sent_data), idx] = sent_data
      mask_tensor[0:len(sent_data), idx] = 1
    batchized_data.append(batch_tensor)
    batchized_mask.append(mask_tensor)

    return batchized_data, batchized_mask

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

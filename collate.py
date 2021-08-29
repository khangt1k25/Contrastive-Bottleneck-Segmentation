from typing import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import collections
from torch._six import string_classes

int_classes = (int, bytes)

def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch
    
    elif isinstance(batch[0], nn.Module):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) if (key.find('idx') < 0 and key != 'T') else [d[key] for d in batch] for key in batch[0] }
        # batch_modified = {key: collate_custom([d[key] for d in batch])  for key in batch[0] if key.find('idx') < 0 }
        return batch_modified
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]
    
    raise TypeError(('Type is {}'.format(type(batch[0]))))
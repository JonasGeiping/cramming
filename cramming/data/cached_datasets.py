"""Write a PyTorch dataset into RAM."""

import torch
import logging

import transformers

log = logging.getLogger(__name__)


def lookup_dtype(vocab_size):
    if vocab_size < 2**8:
        dtype = torch.uint8
    # would really be neat to have uint16 here between the BERT and GPT encoding sizes
    elif vocab_size < 2**16 // 2:
        dtype = torch.int16
    elif vocab_size < 2**32 // 2:
        dtype = torch.int32
    else:
        dtype = torch.int64
    return dtype


class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset into RAM or SDRAM (GPU memory).

    This is only a good idea if you have enough RAM, especially if mapping into SDRAM.
    """

    def __init__(self, dataset, seq_length, vocab_size, num_workers=0, target_device=torch.device("cpu")):
        """Initialize with a given pytorch dataset. The setup dictionary determines cache location and storage type."""
        self.dataset = dataset
        log.info("Caching started ...")
        batch_size = min(len(dataset), 2048)
        cacheloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=transformers.data.data_collator.torch_default_data_collator,
        )
        self.dataset_keys = list(dataset[0].keys())
        seq_lengths = [len(dataset[0][k]) for k in self.dataset_keys]
        assert all([length == seq_lengths[0] for length in seq_lengths])

        # Allocate memory:
        pin = target_device == torch.device("cpu") and torch.cuda.is_available()
        cache_setup = dict(device=target_device, dtype=lookup_dtype(vocab_size), pin_memory=pin)
        self.cache = torch.empty((len(self.dataset), seq_length * 4), **cache_setup)

        pointer = 0
        for data in cacheloader:
            batch_length = data[self.dataset_keys[0]].shape[0]
            data_block = torch.cat([d.to(cache_setup["dtype"]) for d in data.values()], dim=1)
            self.cache[pointer : pointer + batch_length] = data_block
            pointer += batch_length

        self.cache = self.cache.contiguous()
        log.info(f'Dataset sucessfully cached into {"RAM" if target_device == torch.device("cpu") else "SDRAM"}.')

    def __getitem__(self, index):
        """Get sample, target from cache."""
        sample_data_block = self.cache[index]
        sample_dict = dict(zip(self.dataset_keys, torch.chunk(sample_data_block, len(self.dataset_keys), dim=-1)))
        return sample_dict

    def __len__(self):
        """Length is length of self.dataset."""
        return len(self.dataset)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

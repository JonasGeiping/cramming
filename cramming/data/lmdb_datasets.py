"""LMDB dataset to wrap an existing dataset and create a database if necessary."""

import os
import pickle

import platform
import lmdb

import torch
import logging

log = logging.getLogger(__name__)

from .cached_datasets import lookup_dtype
import warnings

warnings.filterwarnings("ignore", "The given buffer is not writable", UserWarning)


class LMDBDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access.

    Originally based on https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
    and
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    but that was several iterations of this file over various projects ago...
    """

    def __init__(self, dataset, cfg_data, cfg_db, path, name="train", can_create=True):
        """Initialize with a given pytorch dataset."""
        if os.path.isfile(os.path.expanduser(path)):
            raise ValueError("LMDB path must lead to a folder containing the databases, not a file.")
        self.dataset = dataset
        self.seq_length = cfg_data.seq_length
        self.dataset_keys = list(dataset[0].keys())
        seq_lengths = [len(dataset[0][k]) for k in self.dataset_keys]
        assert all([length == seq_lengths[0] for length in seq_lengths])

        self.target_dtype = lookup_dtype(cfg_data.vocab_size)

        shuffled = "shuffled" if cfg_db.shuffle_while_writing else ""
        full_name = name + shuffled
        self.path = os.path.join(os.path.expanduser(path), f"{full_name}.lmdb")

        if cfg_db.rebuild_existing_database:
            if os.path.isfile(self.path):
                os.remove(self.path)
                os.remove(self.path + "-lock")

        # Load or create database
        if os.path.isfile(self.path):
            log.info(f"Reusing cached database at {self.path}.")
        else:
            if not can_create:
                raise ValueError(f"No database found at {self.path}. Database creation forbidden in this setting.")
            os.makedirs(os.path.expanduser(path), exist_ok=True)
            log.info(f"Creating database at {self.path}. This may take some time ...")
            create_database(self.dataset, self.path, cfg_data, cfg_db, self.target_dtype)

        # Setup database
        self.cfg = cfg_db
        self.db = lmdb.open(
            self.path,
            subdir=False,
            max_readers=self.cfg.max_readers,
            readonly=True,
            lock=False,
            readahead=self.cfg.readahead,
            meminit=self.cfg.meminit,
            max_spare_txns=self.cfg.max_spare_txns,
        )
        self.access = self.cfg.access

        with self.db.begin(write=False) as txn:
            try:
                self.length = pickle.loads(txn.get(b"__len__"))
                self.keys = pickle.loads(txn.get(b"__keys__"))
            except TypeError:
                raise ValueError(f"The provided LMDB dataset at {self.path} is unfinished or damaged.")

        if self.access == "cursor":
            self._init_cursor()

    def __getstate__(self):
        state = self.__dict__
        state["db"] = None
        if self.access == "cursor":
            state["_txn"] = None
            state["cursor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # Regenerate db handle after pickling:
        self.db = lmdb.open(
            self.path,
            subdir=False,
            max_readers=self.cfg.max_readers,
            readonly=True,
            lock=False,
            readahead=self.cfg.readahead,
            meminit=self.cfg.meminit,
            max_spare_txns=self.cfg.max_spare_txns,
        )
        if self.access == "cursor":
            self._init_cursor()

    def _init_cursor(self):
        """Initialize cursor position."""
        self._txn = self.db.begin(write=False)
        self.cursor = self._txn.cursor()
        self.cursor.first()
        self.internal_index = 0

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

    def __len__(self):
        """Draw length from target dataset."""
        return self.length

    def __getitem__(self, index):
        """Get from database. This is either unordered or cursor access for now.

        Future: Write this class as a proper https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        if self.access == "cursor":
            index_key = "{}".format(index).encode("ascii")
            if index_key != self.cursor.key():
                self.cursor.set_key(index_key)

            byteflow = self.cursor.value()
            self.cursor.next()

        else:
            with self.db.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

        # crime, but ok - we just disabled the warning...:
        # Tested this and the LMDB cannot be corrupted this way, even though byteflow is technically non-writeable
        data_block = torch.frombuffer(byteflow, dtype=self.target_dtype, count=self.seq_length * len(self.dataset_keys))
        sample_dict = dict(zip(self.dataset_keys, torch.chunk(data_block, chunks=len(self.dataset_keys), dim=-1)))
        return sample_dict


def create_database(dataset, database_path, cfg_data, cfg_db, target_dtype):
    """Create an LMDB database from the given pytorch dataset.

    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py

    Removed pyarrow dependency
    but that was several iterations of this file over various projects ago...
    """
    if platform.system() == "Linux":
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        raise ValueError("Provide a reasonable default map_size for your operating system and overwrite this part.")
    db = lmdb.open(
        database_path,
        subdir=False,
        map_size=map_size,
        readonly=False,
        meminit=cfg_db.meminit,
        map_async=True,
    )

    txn = db.begin(write=True)
    idx = 0
    if cfg_db.shuffle_while_writing:
        order = torch.randperm(len(dataset)).tolist()  # this might be a problem for larger dataset sizes?
    else:
        order = range(0, len(dataset))
    for indexing in order:
        data = dataset[indexing]
        # serialize super serially, super slow
        data_block = torch.cat([torch.as_tensor(item, dtype=target_dtype) for item in data.values()], dim=0)
        byteflow = data_block.numpy().tobytes()
        txn.put("{}".format(idx).encode("ascii"), byteflow)
        idx += 1

        if idx % cfg_db.write_frequency == 0:
            log.info(f"[{idx} / {len(dataset)}]")
            txn.commit()
            txn = db.begin(write=True)

    # finalize dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]  # How large will these keys be, too large?
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))
    log.info(f"Database written successfully with {len(keys)} entries.")

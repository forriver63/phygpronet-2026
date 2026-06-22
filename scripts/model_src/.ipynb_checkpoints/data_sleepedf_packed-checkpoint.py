import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_meta(packed_dir: str) -> dict:
    meta_path = os.path.join(packed_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {packed_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class PackedArrays:
    X: np.memmap
    y: np.memmap
    sp: np.memmap
    sw: np.memmap
    sid: np.memmap
    N: int
    L: int
    subjects: List[str]


def open_packed(packed_dir: str, readonly: bool = True) -> PackedArrays:
    meta = load_meta(packed_dir)
    N = int(meta["N"])
    L = int(meta["L"])
    keys = meta["keys"]
    mode = "r" if readonly else "r+"

    X = np.memmap(os.path.join(packed_dir, keys["X"]), dtype=np.float32, mode=mode, shape=(N, 1, L))
    y = np.memmap(os.path.join(packed_dir, keys["y"]), dtype=np.int64,   mode=mode, shape=(N,))
    sp= np.memmap(os.path.join(packed_dir, keys["sp"]),dtype=np.float32, mode=mode, shape=(N,))
    sw= np.memmap(os.path.join(packed_dir, keys["sw"]),dtype=np.float32, mode=mode, shape=(N,))
    sid=np.memmap(os.path.join(packed_dir, keys["sid"]),dtype=np.int32,  mode=mode, shape=(N,))

    subjects = list(meta.get("subjects", []))
    return PackedArrays(X=X, y=y, sp=sp, sw=sw, sid=sid, N=N, L=L, subjects=subjects)


def subject_split_indices(packed: PackedArrays, seed: int = 42,
                          val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subject-wise split: train/val/test indices over samples.
    """
    rng = np.random.RandomState(seed)

    all_sids = np.array(packed.sid, dtype=np.int32)
    uniq = np.unique(all_sids)
    rng.shuffle(uniq)

    nS = len(uniq)
    n_test = max(1, int(round(nS * test_ratio)))
    n_val  = max(1, int(round(nS * val_ratio)))
    test_sids = set(uniq[:n_test].tolist())
    val_sids  = set(uniq[n_test:n_test+n_val].tolist())
    train_sids= set(uniq[n_test+n_val:].tolist())

    idx = np.arange(packed.N, dtype=np.int64)
    tr_idx = idx[np.isin(all_sids, list(train_sids))]
    va_idx = idx[np.isin(all_sids, list(val_sids))]
    te_idx = idx[np.isin(all_sids, list(test_sids))]

    return tr_idx, va_idx, te_idx


class SleepEDFPackedDataset(Dataset):
    """
    Fast dataset on memmap.
    """
    def __init__(self, packed: PackedArrays, indices: np.ndarray,
                 want_mech: bool = True, return_sid: bool = False):
        self.packed = packed
        self.indices = np.asarray(indices, dtype=np.int64)
        self.want_mech = bool(want_mech)
        self.return_sid = bool(return_sid)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])

        x = np.array(self.packed.X[j], dtype=np.float32, copy=False)  # (1,L)
        y = int(self.packed.y[j])

        x_t = torch.from_numpy(x)
        y_t = torch.tensor(y, dtype=torch.long)

        if not self.want_mech:
            if self.return_sid:
                sid = int(self.packed.sid[j])
                return x_t, y_t, torch.tensor(sid, dtype=torch.int32)
            return x_t, y_t

        sp = float(self.packed.sp[j])
        sw = float(self.packed.sw[j])

        sp_t = torch.tensor(sp, dtype=torch.float32)
        sw_t = torch.tensor(sw, dtype=torch.float32)

        if self.return_sid:
            sid = int(self.packed.sid[j])
            sid_t = torch.tensor(sid, dtype=torch.int32)
            return x_t, y_t, sp_t, sw_t, sid_t

        return x_t, y_t, sp_t, sw_t


def make_weighted_sampler_from_indices(packed: PackedArrays, indices: np.ndarray) -> WeightedRandomSampler:
    """
    True weighted sampling but very fast now:
    We read y from memmap in one shot.
    """
    idx = np.asarray(indices, dtype=np.int64)
    y = np.array(packed.y[idx], dtype=np.int64)

    counts = np.bincount(y, minlength=5).astype(np.float64)
    w_class = counts.sum() / (counts + 1e-9)
    weights = w_class[y]

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True
    )

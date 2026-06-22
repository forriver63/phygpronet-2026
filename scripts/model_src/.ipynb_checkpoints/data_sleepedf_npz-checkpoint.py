import os, glob
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

@dataclass
class Batch:
    x30: torch.Tensor  # (B,1,30*sf)
    x90: torch.Tensor  # (B,1,90*sf)
    x5:  torch.Tensor  # (B,1,6,5*sf)
    y:   torch.Tensor  # (B,)
    sid: list          # list[str]

class SleepEDFNpzDataset(Dataset):
    def __init__(self, npz_paths: List[str]):
        self.paths = npz_paths
        self.index = []  # (file_idx, sample_idx)
        self.cache: Dict[int, dict] = {}

        for fi, p in enumerate(self.paths):
            obj = np.load(p, allow_pickle=True)
            n = int(obj["y"].shape[0])
            for si in range(n):
                self.index.append((fi, si))

    def __len__(self):
        return len(self.index)

    def _load(self, fi: int):
        if fi in self.cache:
            return self.cache[fi]
        obj = np.load(self.paths[fi], allow_pickle=True)
        sid = str(obj.get("subject", obj.get("sid", os.path.basename(self.paths[fi])[:6])))
        d = {
            "x30": obj["x30"],
            "x90": obj["x90"],
            "x5":  obj["x5"],
            "y":   obj["y"],
            "sid": sid
        }
        self.cache[fi] = d
        return d

    def __getitem__(self, idx: int):
        fi, si = self.index[idx]
        d = self._load(fi)
        x30 = torch.from_numpy(d["x30"][si])  # (1,T)
        x90 = torch.from_numpy(d["x90"][si])  # (1,3T)
        x5  = torch.from_numpy(d["x5"][si])   # (1,6,t5)
        y   = torch.tensor(int(d["y"][si]), dtype=torch.long)
        return x30, x90, x5, y, d["sid"]

def list_npz(npz_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

def subject_split(npz_paths: List[str], seed=42, ratios=(0.8,0.1,0.1)):
    rng = np.random.RandomState(seed)
    paths = np.array(npz_paths, dtype=object)
    rng.shuffle(paths)
    n = len(paths)
    ntr = int(n * ratios[0])
    nva = int(n * ratios[1])
    train = paths[:ntr].tolist()
    val   = paths[ntr:ntr+nva].tolist()
    test  = paths[ntr+nva:].tolist()
    return train, val, test

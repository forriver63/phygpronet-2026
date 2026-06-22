import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_meta(packed_dir: str) -> dict:
    meta_path = os.path.join(packed_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {packed_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta"]


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
    bands: Optional[Dict[str, np.ndarray]] = None  # raw band powers


def open_packed(packed_dir: str, readonly: bool = True, load_bands: bool = False) -> PackedArrays:
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

    bands = None
    if load_bands:
        bp = os.path.join(packed_dir, "bands_raw.npz")
        if os.path.exists(bp):
            with np.load(bp) as f:
                bands = {k: np.asarray(f[k], dtype=np.float32) for k in BAND_ORDER}
    return PackedArrays(X=X, y=y, sp=sp, sw=sw, sid=sid, N=N, L=L, subjects=subjects, bands=bands)


def subject_split_indices(packed: PackedArrays, seed: int = 42,
                          val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Modes (controlled at construction):
      want_mech=True (default): legacy returns (x, y, sp, sw)
      multi_band=True: returns (x, y, bands_5) where bands_5 is (5,) float tensor in band order
                       (delta, theta, alpha, sigma, beta) z-scored using `band_stats`.
      per_recording_norm=True: applies x = (x - mu_sid) / sd_sid using `rec_stats`.

    multi_band and want_mech are mutually exclusive on output; if both set, multi_band wins.
    """
    def __init__(self, packed: PackedArrays, indices: np.ndarray,
                 want_mech: bool = True, return_sid: bool = False,
                 multi_band: bool = False,
                 band_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 per_recording_norm: bool = False,
                 rec_stats: Optional[Dict[str, Dict[str, float]]] = None):
        self.packed = packed
        self.indices = np.asarray(indices, dtype=np.int64)
        self.want_mech = bool(want_mech)
        self.return_sid = bool(return_sid)
        self.multi_band = bool(multi_band)
        self.per_recording_norm = bool(per_recording_norm)

        if self.multi_band:
            assert packed.bands is not None, "multi_band requires open_packed(..., load_bands=True)"
            assert band_stats is not None, "multi_band requires band_stats"
            # Pre-build a (N, 5) z-scored matrix once, in memory (4 bytes * N * 5 ~ 8MB for 414k -> fine)
            stacks = []
            for k in BAND_ORDER:
                v = packed.bands[k]
                mu = float(band_stats[k]["mean"]); sd = float(band_stats[k]["std"])
                stacks.append(((v - mu) / sd).astype(np.float32))
            self.band_z = np.stack(stacks, axis=1)  # (N, 5)
        else:
            self.band_z = None

        if self.per_recording_norm:
            assert rec_stats is not None, "per_recording_norm requires rec_stats"
            # Build dense lookup keyed by sid integer
            sid_arr = np.array(packed.sid, dtype=np.int32)
            max_sid = int(sid_arr.max()) + 1
            mu_lut = np.zeros(max_sid, dtype=np.float32)
            sd_lut = np.ones(max_sid, dtype=np.float32)
            for k, v in rec_stats.items():
                s = int(k)
                if 0 <= s < max_sid:
                    mu_lut[s] = float(v["mean"])
                    sd_lut[s] = float(v["std"])
            self.mu_lut = mu_lut
            self.sd_lut = sd_lut
        else:
            self.mu_lut = None
            self.sd_lut = None

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])

        x = np.array(self.packed.X[j], dtype=np.float32, copy=True)  # (1,L)

        if self.per_recording_norm:
            s = int(self.packed.sid[j])
            mu = float(self.mu_lut[s]); sd = float(self.sd_lut[s])
            x = (x - mu) / sd

        y = int(self.packed.y[j])
        x_t = torch.from_numpy(x)
        y_t = torch.tensor(y, dtype=torch.long)

        if self.multi_band:
            bz = self.band_z[j]  # (5,)
            bz_t = torch.from_numpy(bz.copy())
            if self.return_sid:
                sid = int(self.packed.sid[j])
                return x_t, y_t, bz_t, torch.tensor(sid, dtype=torch.int32)
            return x_t, y_t, bz_t

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
            return x_t, y_t, sp_t, sw_t, torch.tensor(sid, dtype=torch.int32)
        return x_t, y_t, sp_t, sw_t


def make_weighted_sampler_from_indices(packed: PackedArrays, indices: np.ndarray) -> WeightedRandomSampler:
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

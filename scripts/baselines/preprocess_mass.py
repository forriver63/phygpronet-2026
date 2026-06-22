"""Pre-process MASS to compact format: channel 20, 100Hz, float32. Saves ~100x size reduction."""
import os, glob, numpy as np
from scipy.signal import resample_poly

src_dir = "/root/autodl-tmp/data/mass_npz"
out_dir = "/root/autodl-tmp/data/mass_npz_small"
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
for fi, f in enumerate(files):
    n = os.path.basename(f)
    o = os.path.join(out_dir, n)
    if os.path.exists(o): continue
    print(f"[{fi+1}/{len(files)}] {n}")
    with np.load(f) as z:
        x = z["data"][:, 20, :].astype(np.float32)   # (N, 7680) at 256Hz
        y = z["labels"][:]
    x = resample_poly(x, 100, 256, axis=-1).astype(np.float32)  # (N, 3000)
    if x.shape[1] != 3000:
        x = x[:, :3000] if x.shape[1] > 3000 else np.pad(x, ((0,0),(0,3000-x.shape[1])))
    np.savez_compressed(o, data=x, labels=y)
print("done")

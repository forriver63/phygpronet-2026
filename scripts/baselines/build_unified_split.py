"""Build unified subject-level 5-fold CV split for Sleep-EDF-78.

Convention follows TinySleepNet v2:
  - 78 subjects (recording-level, each SC4001/SC4002/... is one unit)
  - rng = RandomState(42); perm = rng.permutation(subjects); fold_subjects = array_split(perm, 5)
  - 10% of train as val: rng2 = RandomState(100+fold).choice(train, n_val, replace=False)
"""
import argparse, glob, json, os, re
import numpy as np


def subject_id_from_path(p):
    """Group by first 5 chars (matches TinySleepNet v2 convention).
    SC4001E0.npz -> "SC400", SC4002F0.npz -> "SC400" (paired two-night subject).
    """
    return os.path.basename(p)[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir of *.npz Sleep-EDF files (used to enumerate subjects)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    subj_map = {}
    for f in files:
        sid = subject_id_from_path(f)
        subj_map.setdefault(sid, []).append(os.path.basename(f))
    subjects = sorted(subj_map.keys())
    print(f"Found {len(subjects)} subjects, {len(files)} npz files")

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(subjects)
    fold_subjects = np.array_split(perm, 5)

    out = {
        "version": "sleepedf_subject_5fold_seed42_v1",
        "convention": "78-recording-level subjects, RandomState(42).permutation + array_split(5); val=10% of train via RandomState(100+fold)",
        "seed": args.seed,
        "n_subjects": len(subjects),
        "n_files": len(files),
        "folds": {},
    }

    for fold in range(5):
        test_subj = sorted(fold_subjects[fold].tolist())
        train_pool = [s for s in subjects if s not in set(test_subj)]
        n_val = max(1, round(len(train_pool) * 0.10))
        rng2 = np.random.RandomState(100 + fold)
        val_idx = rng2.choice(len(train_pool), size=n_val, replace=False)
        val_subj = sorted([train_pool[i] for i in val_idx])
        train_subj = sorted([s for s in train_pool if s not in set(val_subj)])

        # sanity check
        assert set(train_subj).isdisjoint(set(val_subj))
        assert set(train_subj).isdisjoint(set(test_subj))
        assert set(val_subj).isdisjoint(set(test_subj))
        assert set(train_subj) | set(val_subj) | set(test_subj) == set(subjects)

        out["folds"][str(fold)] = {
            "train": train_subj,
            "val": val_subj,
            "test": test_subj,
            "n_train": len(train_subj),
            "n_val": len(val_subj),
            "n_test": len(test_subj),
        }
        print(f"  Fold {fold}: train={len(train_subj)} val={len(val_subj)} test={len(test_subj)} "
              f"(test subjects: {test_subj[:5]}...)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

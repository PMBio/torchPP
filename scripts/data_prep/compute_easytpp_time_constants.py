"""EASYTPP_MEAN_IET / EASYTPP_TIME_SCALE of an EasyTPP train split (torchpp/data/easytpp_data.py).

MEAN_IET = mean inter-event time after removing each sequence's offset (pooled over all train gaps);
TIME_SCALE = mean sequence duration after the mean-IET shift, t_last - t0 + MEAN_IET. Reproduces the
library defaults for taxi, amazon and taobao (Google Drive release) and gives the easytpp_hf_f64/taobao override.

    python scripts/data_prep/compute_easytpp_time_constants.py $TORCHPPDATA/easytpp_hf_f64/taobao/train.pt
"""
import sys

import numpy as np
import torch

for p in sys.argv[1:]:
    seqs = torch.load(p, weights_only=False)
    rel = [s.time_points.double().numpy() - float(s.time_points[0]) for s in seqs]
    iet = np.concatenate([np.diff(r) for r in rel]).mean()
    scale = np.mean([r[-1] + iet for r in rel])
    print(f"{p}: n_seq={len(seqs)} MEAN_IET={iet:.6f} TIME_SCALE={scale:.6f}")

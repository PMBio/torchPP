"""Collect calibration_raw.json files from run_calibration.py into one CSV (the input of the calibration table).

    python paper/calibration/make_report.py calibration_out/calibration_raw.json [...] --out paper/results/calibration.csv
"""

import argparse
import csv
import json

FIELDS = ["n_events", "tie_fraction", "ks_D", "mean_lambda", "var_lambda", "top_label_ece", "classwise_ece",
          "acc_at_t"]
DATASET_NAMES = {"mimic_iv_icu": "mimic_iv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw", nargs="+", help="calibration_raw.json files")
    ap.add_argument("--out", default="calibration.csv")
    args = ap.parse_args()

    rows = []
    for path in args.raw:
        raw = json.load(open(path))
        for f in raw["failures"]:
            print("failure:", f)
        for r in raw["results"]:
            rows.append(dict(model_type=r["model"], dataset=DATASET_NAMES.get(r["dataset"], r["dataset"]),
                             seed=r["rep"], **{k: r[k] for k in FIELDS}))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()

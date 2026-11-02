# UKB CSV Conversion

UK Biobank data requires an approved UK Biobank application. No UKB data, derived
data or UKB-trained models are distributed with this repository; only the
conversion code is.

`scripts/data_prep/prepare_ukb_cluster_dataset.py` converts a UKB field export into
the binary format that `torchpp.data.ukb_data.UKBDataset` loads. The token
vocabulary (`labels.csv`) and the ICD-10 mapping (`icd10_codes_mod.tsv`) follow the
[Delphi repository](https://github.com/gerstung-lab/Delphi); pass their paths on the
command line.

## Binary contract

- Raw `.bin` rows are `uint32` triples `[batch_id, time_days_since_dob, token_id]`.
- Batch IDs are globally contiguous across splits.
- Splits are `70/20/10` for `train.bin`, `val.bin`, `test.bin`.
- Token IDs match the line numbers in `labels.csv`:
  - `2` = female
  - `3` = male
  - `4..12` = lifestyle covariates
  - `13..1268` = ICD-derived event types
  - `1269` = death

## Field naming

UKB exports name fields either `f.31.0.0` or `31-0.0`. The mapping is mechanical:

1. Drop the leading `f.`.
2. Replace the first remaining dot with `-`.
3. Keep the last dot between instance and array index.

The script accepts both forms.

## What the script does

- resolves either field naming scheme,
- infers the sex-code mapping from the raw export,
- converts date columns to days since month-level date of birth,
- emits token IDs that match `labels.csv`,
- writes `train.bin`, `val.bin`, `test.bin` and a `summary.json`.

```bash
python scripts/data_prep/prepare_ukb_cluster_dataset.py \
  --csv-path <ukb_export.csv> --labels-file <Delphi>/labels.csv \
  --icd-map-file <Delphi>/icd10_codes_mod.tsv --out-dir $TORCHPPDATA/ukb_real_data
```

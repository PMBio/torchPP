# integration/

Full-train and cross-library equivalence checks live **outside pytest** — they
train models end to end and are too slow for the unit/contract/models gate.

- **EasyTPP equivalence harness:** `scripts/equivalence/` (run manually).
  It checks the native IFTPP / NHP / AttNHP ports against EasyTPP.

Nothing here is collected by pytest. Keep it that way: per-family numerical
CI-value regressions belong in `tests/models/`; this directory only documents
where the heavier validation lives.

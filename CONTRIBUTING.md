# Contributing

Bug reports, questions and pull requests are welcome.

## Development install

```bash
git clone https://github.com/PMBio/torchPP.git torchpp && cd torchpp
uv sync --all-extras --group dev
```

## Tests

There is no CI. Please run the tests that cover your change before you open a pull request.

```bash
pytest tests/unit tests/contract -q                          # fast tier
pytest tests/models/test_terminating_values.py -k my_model   # one model-tier file, scoped
```

The model tests run over every registered model, so the full `tests/models/` tier is
slow and some cells need several GB of memory. We recommend scoping it with `-k` and running
one file at a time. On a laptop, `-n 2` (pytest-xdist) gives enough parallelism.

## Adding a model or a dataset

A new model needs one registry entry, a test configuration and a preset. The parametrized
tests then cover it automatically. See [docs/extending.md](docs/extending.md) for details.

## Style

Keep comments short and instructive. They should describe what the code does now, not how it
got there. Public functions should have a docstring that states the argument shapes.

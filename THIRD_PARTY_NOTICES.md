# Third-party notices

TorchPP is released under the MIT License (`LICENSE`). Parts of it are adapted from the projects below
and remain under their original licenses.

## EasyTPP

- Source: https://github.com/ant-research/EasyTemporalPointProcess
- License: Apache License 2.0 (`LICENSES/Apache-2.0.txt`)
- Copyright 2022 The EasyTPP Authors. All rights reserved.
- EasyTPP NOTICE: "EasyTPP is a open source tool developed by Machine Intelligence Team.
  Copyright (c) 2020-2022, Ant Group Holding Limited. Licensed under the Apache License, Version 2.0."

Adapted and modified for TorchPP:

| TorchPP file | Adapted from (EasyTPP) |
|---|---|
| `torchpp/modules/nn/attnhp.py` | `torch_attnhp.py`, `torch_baselayer.py` |
| `torchpp/modules/nn/encoder.py` (`ContTimeLSTMCell`) | `torch_nhp.py` |
| `torchpp/modules/nn/lognormal_mixture.py` | `torch_intensity_free.py` |
| `torchpp/modules/nn/transformer.py` (THP and SAHP intensity heads) | `torch_thp.py`, `torch_sahp.py` |

EasyTPP's implementations of these models follow the original authors' code: NHP (Mei & Eisner, 2017),
AttNHP (Yang et al., 2022), IntensityFree (Shchur et al., 2020), THP (Zuo et al., 2020) and SAHP
(Zhang et al., 2020).

## nanoGPT and Delphi

The transformer encoder in `torchpp/modules/nn/transformer.py` is adapted from Delphi, which builds on
nanoGPT.

- nanoGPT: https://github.com/karpathy/nanoGPT. Copyright (c) 2022 Andrej Karpathy. MIT License.
- Delphi: https://github.com/gerstung-lab/Delphi. Copyright (c) 2024 Gerstung Lab. MIT License
  (code only).

MIT License text (applies to both):

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

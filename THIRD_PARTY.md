# Third-party code and methods

The baselines and one predictor of DiscoRD re-implement or adapt published methods. We keep
the original attributions below; the corresponding papers are cited in the paper.

| File | Origin | Licence |
|---|---|---|
| `src/discordrd/sparseeval_mlp.py` | Re-implementation of the SparseEval MLP estimator and its anchor selection (Zhang et al., 2026, ICLR), following the paper and the released code; trained on the memory only, with the checkpoint chosen on the validation set | written for this repository |
| `src/discordrd/baselines/krr.py` | mRMR (MIQ) selection and degree-2 polynomial kernel ridge regression of Random+KRR and mRMR+KRR (Bowyer et al., 2026), following the released benchmark-prediction code | MIT (see below) |
| `src/discordrd/baselines/mrmr_continuous.py` | Adapted from the released benchmark-prediction code (MIT) | MIT (see below) |
| `src/discordrd/baselines/irt.py` | Nested MIRT re-implemented from the equations of Krumdick et al. (2026); 1PL helpers | written for this repository |
| `src/discordrd/baselines/estimators.py` | tinyBenchmarks-IRT / -pIRT / -gpIRT (Maia Polo et al., 2024; released code under MIT), IRSL-CAT-1PL (Truong et al., 2026), BenchPress-style matrix completion (Zeng et al., 2026), anchor mean and AIPW (Zhang et al., 2026), re-implemented from the papers | written for this repository |

## MIT licence of the benchmark-prediction code

```
MIT License

Copyright (c) 2024 Guanhua Zhang and Moritz Hardt

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

# MiniGPT on TinyStories

DASC7606 graduate course project. This repository records matched-budget experiments on a 13.9M-parameter decoder-only Transformer trained from random initialization. The course-provided MiniGPT implementation derives from nanoGPT under the included MIT license. See `core/README.md` and `extension/README.md` for code provenance and reproduction commands.

## Experiments

All runs used the fixed TinyStories training split, seed 1337, and 2,500 AdamW updates (10.24M processed targets per run). Training used BF16 on one RTX 4060 Laptop GPU. Checkpoints were scored in FP32 on all 389,048 public validation targets.

| Setting | Parameters | Train time (min) | Validation CE | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Six-layer baseline | 13,891,584 | 11.12 | 2.3006 | 9.980 |
| Four layers | 10,342,656 | 8.23 | 2.3401 | 10.383 |
| Half the peak learning rate | 13,891,584 | 11.13 | 2.4554 | 11.651 |
| RMSNorm | 13,886,592 | 11.07 | 2.3045 | 10.019 |

Reducing depth cut training time by 26.0% at a cost of 0.0395 nats in validation CE. RMSNorm stayed within 0.0039 nats of baseline on the full validation split. Its measured time difference was 3.4 seconds, which does not establish a repeatable speedup from one pair of runs.

## Files and reproduction

- [Core experiment report](reports/report.pdf) and [RMSNorm extension report](reports/extension_report.pdf)
- [Core code, tests and configurations](core/) and [extension code and configuration](extension/)
- [Recorded core run evidence](core/runs/) and [RMSNorm run evidence](extension/results/rmsnorm/)

Use Python 3.13 and PyTorch 2.10.0 with a compatible CUDA build. Obtain the fixed dataset ZIP from the course release and follow the README for each experiment. The package includes training logs, measured metrics, loss curves and fixed-prompt samples. Checkpoints and the course dataset are distributed separately from this repository.


# Optional extension — RMSNorm in MiniGPT

This extension compares the core six-layer LayerNorm baseline with one matched six-layer run that uses RMSNorm. At every pre-normalization site and at the final normalization, `torch.nn.RMSNorm` divides by the root mean square of the input channels and applies a learned scale. The fixed epsilon is 1e-5. The hypothesis was that removing mean-centering and affine offsets could reduce time per update while keeping validation cross-entropy close to the baseline. GPU kernel selection may affect timing independently of the arithmetic.

The only change in the model source replaces the 13 LayerNorm modules with RMSNorm. The packaged configuration matches the core baseline values. Both runs use the unchanged TinyStories train split, vocabulary, 256-token context, seed 1337, BF16 training and 10,240,000 processed targets. The source under `code/` derives from the supplied MiniGPT implementation, which credits nanoGPT under the included MIT license. The normalization method follows Biao Zhang and Rico Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (NeurIPS 2019).

## Contents

- `extension_report.pdf` gives the quantitative comparison, validation curves and interpretation.
- `code/src` and `code/scripts/plot_run.py` reproduce the extension.
- `configs/rmsnorm.yaml` records the exact run configuration.
- `results/rmsnorm` holds the run metrics, learning curve, generations and full public validation results.

## Reproduce

Use Windows with Python 3.13.9, PyTorch 2.10.0+cu126 and the other pins in `code/requirements.txt`. The original run used a GeForce RTX 4060 Laptop GPU (8 GiB, driver 537.53). Extract the course-provided fixed data ZIP at the root of the extension directory for local reproduction. This creates `data/train.bin`, `data/val.bin`, `data/tokenizer.json` and the manifest.

From the extension root, with its virtual environment active:

```powershell
Set-Location code
$run = "../outputs/rmsnorm"
$data = "../data"
python -m src.train --config ../configs/rmsnorm.yaml --out_dir $run --data-dir $data --device cuda
python scripts/plot_run.py $run
python -m src.sample --checkpoint "$run/best.pt" --tokenizer "$data/tokenizer.json" --out "$run/samples.json" --device cuda
python -m src.evaluate --checkpoint "$run/best.pt" --tokens "$data/val.bin" --tokenizer "$data/tokenizer.json" --max-tokens 0 --out "$run/evaluation.json" --device cuda
```

Full evaluation scores all 389,048 target tokens in FP32. The training curve uses the sampled mixed-precision validation protocol.



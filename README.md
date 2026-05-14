<div align="center">

<h2>OmniDoc-TokenBench</h2>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HF Dataset](https://img.shields.io/badge/HF%20Dataset-OmniDoc--TokenBench-yellow?logo=huggingface)](https://huggingface.co/datasets/alibabagroup/OmniDoc-TokenBench)
[![arXiv](https://img.shields.io/badge/arXiv-QwenImageVAE2.0-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.13565)

</div>

---

## 📄 Overview

### Introduction

> 🤗 **Dataset Download**: [https://huggingface.co/datasets/alibabagroup/OmniDoc-TokenBench](https://huggingface.co/datasets/alibabagroup/OmniDoc-TokenBench)

We propose **OmniDoc-TokenBench** in [Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565), a curated benchmark specifically designed to evaluate VAE reconstruction on text-rich document images. It contains ~3K samples spanning nine categories (*book*, *slides*, *color textbook*, *exam paper*, *academic paper*, *magazine*, *financial report*, *newspaper*, *note*) in both English and Chinese, alongside an evaluation toolkit supporting PSNR, SSIM, LPIPS, FID, and OCR-based NED metrics.

<p align="center">
  <img src="assets/bench.png" alt="OmniDoc-TokenBench" width="80%" />
<p>

We develop OmniDoc-TokenBench based on [OmniDocBench](https://github.com/opendatalab/OmniDocBench). First, we crop each sample from a text block and resize it to 256×256, then filter for a character count range ([200, 600] for Chinese, [300, 600] for English) to ensure a reference font size of approximately 16px and 10px, respectively. Finally, we deduplicate via n-gram overlap and manually inspect for quality.


### Evaluation Metric

Beyond traditional metrics (PSNR, SSIM, LPIPS, FID), we use **NED** (Normalized Edit Distance) as the primary text-fidelity metric. NED directly measures text preservation by comparing recognized character sequences between original and reconstructed images using Levenshtein distance:

$$
\mathrm{NED} = \frac{1}{N}\sum_{i=1}^{N}\left(1 - \frac{d_{\mathrm{edit}}(s_{\mathrm{gt}}^{(i)}, s_{\mathrm{recon}}^{(i)})}{\max(|s_{\mathrm{gt}}^{(i)}|, |s_{\mathrm{recon}}^{(i)}|)}\right)
$$

NED is sensitive to semantic corruption such as character substitutions, making it a necessary complementary metric when traditional metrics alone are insufficient.

---

## 📊 Performance

We conduct a comprehensive evaluation on OmniDoc-TokenBench (~3K text-rich images, 256×256 resolution). Models are grouped by spatial compression factor and sorted by NED within each group.

<p align="center">
  <img src="assets/results.png" alt="Eval-Results" width="65%" />
</p>

Our Qwen-Image-VAE-2.0 achieves state-of-the-art reconstruction across all compression ratios. The f16c128 variant attains SSIM **0.9706** and PSNR **30.45 dB**, surpassing the best f8 baseline (FLUX.1-dev at 0.9364 / 26.24 dB) despite 2× higher spatial compression. In terms of text fidelity (NED), f16c128 reaches **0.9617**, exceeding all evaluated VAEs. Even under extreme f32 compression, our f32c192 achieves NED **0.8555**, surpassing multiple f16 baselines.

---

## ⚡ Evaluation

### Installation

```bash
git clone https://github.com/alibaba/OmniDoc-TokenBench.git
cd OmniDoc-TokenBench

pip install torch torchvision piq lpips pytorch-fid pillow numpy tqdm
pip install paddleocr python-Levenshtein  # required for NED
```

### Download the Dataset

```bash
# Make sure the hf CLI is installed
curl -LsSf https://hf.co/cli/install.sh | bash
# Download the dataset
hf download alibabagroup/OmniDoc-TokenBench --repo-type=dataset --local-dir ./dataset_download
# Move benchmark images to project root as gt_dir
mv ./dataset_download/data ./gt_dir && rm -rf ./dataset_download
```

### Reconstruct Images with Your VAE

We provide an example reconstruction script at the project root. Edit `example_recon.py` to set your VAE model path, then run:

```bash
python example_recon.py  # FLUX1.dev as example
```

### Compute Metrics

Place your ground-truth images in `gt_dir/` and reconstructed images in `recon_dir/` (filenames must match one-to-one).

```bash
# Compute NED only (default)
python eval_metrics.py --gt_dir ./gt_dir --recon_dir ./recon_dir

# Compute traditional metrics (PSNR / SSIM / LPIPS / FID)
python eval_metrics.py --gt_dir ./gt_dir --recon_dir ./recon_dir --mode pixel

# Compute all metrics
python eval_metrics.py --gt_dir ./gt_dir --recon_dir ./recon_dir --mode all

# Specify output directory and device
python eval_metrics.py --gt_dir ./gt_dir --recon_dir ./recon_dir --save_path ./results --device cuda
```

### Output

The script writes results (FLUX1.dev as example) to the `--save_path` directory (default: `./eval_results`):

- `results.json` --- Aggregated metrics:
  ```json
  {
    "num_samples": 3042,
    "PSNR": 26.2377,
    "SSIM": 0.9364,
    "LPIPS": 0.0247,
    "FID": 0.5543,
    "NED": 0.9546
  }
  ```

- `ned_details.json` --- Per-image OCR results and NED scores (generated when `--mode` is `ned` or `all`):
  ```json
  {
    "avg_ned": 0.9546,
    "total_samples": 3042,
    "valid_samples": 3042,
    "details": [
      {
        "file": "0001.png",
        "gt_ocr": "ocr output from gt image...",
        "recon_ocr": "ocr output from recon image...",
        "ned": 0.9764
      }
    ]
  }
  ```

### Notes

- `FID` and `LPIPS` require downloading model checkpoints on the first run (InceptionV3 ~90MB for FID, VGG16 ~530MB for LPIPS). Ensure network access or pre-download the weight files.
- PaddleOCR defaults to CPU inference. For large-scale evaluation, consider switching to GPU by setting `device="gpu"` in `compute_ned()`.
- The progress bar for PSNR/SSIM/LPIPS displays running means in real time.

---

## 📝 Citation

If you use OmniDoc-TokenBench or this evaluation toolkit in your research, please cite:

```bibtex
@misc{zhang2026qwenimagevae20technicalreport,
      title={Qwen-Image-VAE-2.0 Technical Report}, 
      author={Zekai Zhang and Deqing Li and Kuan Cao and Yujia Wu and Chenfei Wu and Yu Wu and Liang Peng and Hao Meng and Jiahao Li and Jie Zhang and Kaiyuan Gao and Kun Yan and Lihan Jiang and Ningyuan Tang and Shengming Yin and Tianhe Wu and Xiao Xu and Xiaoyue Chen and Yan Shu and Yanran Zhang and Yilei Chen and Yixian Xu and Yuxiang Chen and Zhendong Wang and Zihao Liu and Zikai Zhou and Yiliang Gu and Yi Wang and Xiaoxiao Xu and Lin Qu},
      year={2026},
      eprint={2605.13565},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.13565}, 
}
```

---

##  Acknowledgements

OmniDoc-TokenBench is a derivative dataset based on [OmniDocBench](https://github.com/opendatalab/OmniDocBench), Thanks for their great work.

## License

This dataset is developed by the Qwen Team at Alibaba Group, and licensed under the [Apache License 2.0](LICENSE).

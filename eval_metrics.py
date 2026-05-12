# Copyright 2026 Qwen Team, Alibaba Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     python eval_metrics.py --gt_dir ./gt --recon_dir ./recon                    # NED only
#     python eval_metrics.py --gt_dir ./gt --recon_dir ./recon --mode pixel       # PSNR/SSIM/LPIPS/FID
#     python eval_metrics.py --gt_dir ./gt --recon_dir ./recon --mode all         # All metrics
#
# Dependencies:
#     pip install torch torchvision piq lpips pytorch-fid pillow numpy tqdm
#     pip install paddleocr python-Levenshtein  # for NED

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from piq import psnr, ssim
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def get_image_list(gt_dir: Path) -> list[str]:
    files = sorted(f.name for f in gt_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    assert files, f"No images found in {gt_dir}"
    return files


@torch.no_grad()
def compute_pixel_metrics(gt_dir: Path, recon_dir: Path, files: list[str], device: torch.device):
    lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()

    psnr_list, ssim_list, lpips_list = [], [], []

    pbar = tqdm(files, desc="PSNR/SSIM/LPIPS")
    for name in pbar:
        gt = Image.open(gt_dir / name).convert("RGB")
        rec = Image.open(recon_dir / name).convert("RGB")
        if gt.size != rec.size:
            rec = rec.resize(gt.size, Image.Resampling.BICUBIC)

        gt_t = to_tensor(gt).unsqueeze(0).to(device)  # [1,3,H,W] ∈ [0,1]
        rec_t = to_tensor(rec).unsqueeze(0).to(device)

        p = psnr(rec_t, gt_t, data_range=1.0).item()
        if np.isfinite(p):
            psnr_list.append(p)
        ssim_list.append(ssim(rec_t, gt_t, data_range=1.0, downsample=False).item())

        lpips_list.append(lpips_fn(rec_t * 2 - 1, gt_t * 2 - 1).item())

        pbar.set_postfix(
            PSNR=f"{np.mean(psnr_list):.2f}",
            SSIM=f"{np.mean(ssim_list):.4f}",
            LPIPS=f"{np.mean(lpips_list):.4f}",
        )

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


def compute_fid(gt_dir: Path, recon_dir: Path, device: torch.device, batch_size: int = 32) -> float:
    return calculate_fid_given_paths(
        [str(gt_dir), str(recon_dir)],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )


def compute_ned(gt_dir: Path, recon_dir: Path, files: list[str]) -> tuple[float, int, list[dict]]:
    from paddleocr import PaddleOCR
    import Levenshtein

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )

    def extract_text(img_path: Path) -> str:
        img = np.array(Image.open(img_path).convert("RGB"))
        res = ocr.predict(np.ascontiguousarray(img))
        if res and len(res) > 0 and isinstance(res[0], dict):
            return "".join(res[0].get("rec_texts", []))
        return ""

    ned_list = []
    details = []
    pbar = tqdm(files, desc="NED (OCR)")
    for name in pbar:
        text_gt = extract_text(gt_dir / name)
        if not text_gt:
            continue
        text_rec = extract_text(recon_dir / name)
        max_len = max(len(text_gt), len(text_rec))
        ned = 1.0 - Levenshtein.distance(text_gt, text_rec) / max_len
        ned_list.append(ned)

        detail = {
            "file": name,
            "gt_text": text_gt,
            "rec_text": text_rec,
            "ned": round(ned, 4),
        }
        details.append(detail)

        print(f"GT : {text_gt}")
        print(f"REC: {text_rec}")
        print(f"NED: {ned:.4f}")
        print()

    return (np.mean(ned_list) if ned_list else 0.0), len(ned_list), details


def main():
    parser = argparse.ArgumentParser(description="Image reconstruction evaluation")
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--recon_dir", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, default=Path("./eval_results"))
    parser.add_argument("--mode", type=str, default="ned", choices=["ned", "pixel", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    files = get_image_list(args.gt_dir)
    print(f"Found {len(files)} images | Device: {device} | Mode: {args.mode}")

    results = {"num_samples": len(files)}

    if args.mode in ("pixel", "all"):
        avg_psnr, avg_ssim, avg_lpips = compute_pixel_metrics(args.gt_dir, args.recon_dir, files, device)
        fid = compute_fid(args.gt_dir, args.recon_dir, device)
        results.update({
            "PSNR": round(avg_psnr, 4),
            "SSIM": round(avg_ssim, 4),
            "LPIPS": round(avg_lpips, 4),
            "FID": round(fid, 4),
        })

    ned_details = []
    if args.mode in ("ned", "all"):
        avg_ned, ned_count, ned_details = compute_ned(args.gt_dir, args.recon_dir, files)
        results.update({
            "NED": round(avg_ned, 4),
            "NED_samples": ned_count,
        })

    print("\n" + "=" * 36)
    for k, v in results.items():
        print(f"  {k:>12}: {v}")
    print("=" * 36)

    args.save_path.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    if ned_details:
        ned_output = {
            "avg_ned": round(results.get("NED", 0.0), 4),
            "total_samples": len(files),
            "valid_samples": len(ned_details),
            "details": ned_details,
        }
        with open(args.save_path / "ned_details.json", "w", encoding="utf-8") as f:
            json.dump(ned_output, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {args.save_path / 'results.json'}")
        print(f"Saved → {args.save_path / 'ned_details.json'}")


if __name__ == "__main__":
    main()

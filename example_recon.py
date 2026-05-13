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

import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor
from diffusers import AutoencoderKL
from tqdm import tqdm

# Load your vae, FLUX.1-dev as example
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=torch.float32).cuda().eval()

gt_dir = Path("./gt_png")
recon_dir = Path("./recon_dir")
recon_dir.mkdir(exist_ok=True)

for img_path in tqdm(sorted(gt_dir.glob("*.png")), desc="Reconstructing"):
    img = Image.open(img_path).convert("RGB")
    x = to_tensor(img).unsqueeze(0).cuda() * 2 - 1  # [0,1] -> [-1,1]

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.mode()
        recon = vae.decode(latent).sample

    recon = recon.squeeze(0).clamp(-1, 1)
    recon = ((recon + 1) / 2 * 255).round().to(torch.uint8)
    recon_np = recon.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(recon_np).save(recon_dir / img_path.name)

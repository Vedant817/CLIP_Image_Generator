import os
import argparse
from typing import List, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from transformers import CLIPProcessor, CLIPModel

# BigGAN
try:
    from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names, convert_to_images
except Exception as e:
    raise ImportError(
        "pytorch_pretrained_biggan is required. Install with:\n"
        "  pip install pytorch-pretrained-biggan\n\n"
        "If install fails, try:\n"
        "  pip install git+https://github.com/huggingface/pytorch-pretrained-BigGAN.git\n\n"
        "Original error: " + str(e)
    )

def sanitize_filename(s: str) -> str:
    keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in s)[:200].strip().replace(" ", "_")


class CLIPGuidedBigGAN:
    def __init__(self,
                 device: torch.device = None,
                 biggan_model_name: str = "biggan-deep-256",
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 truncation: float = 0.4):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"[INFO] Using device: {self.device}")

        # Load CLIP (HuggingFace)
        print("[INFO] Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Load BigGAN
        print("[INFO] Loading BigGAN model...")
        self.biggan = BigGAN.from_pretrained(biggan_model_name).to(self.device)
        self.truncation = truncation

        # BigGAN expects float vectors with shape (1, 128)
        self.dim_z = 128

    def generate_biggan_image_from_z(self, z_tensor: torch.Tensor, class_vector: Optional[torch.Tensor] = None):
        """
        z_tensor: torch tensor shape (1, 128) on device, values ~ N(0, 1)
        class_vector: one-hot vector shape (1, 1000) or None (defaults to uniform/zeros)
        returns: PIL Image (RGB)
        """
        with torch.no_grad():
            # BigGAN forward expects (z, class_vector, truncation)
            if class_vector is None:
                # use zeros (no class) — BigGAN requires class vector, so use a zero vector (effectively "no class")
                class_vector = torch.zeros((1, 1000), device=self.device)
            images = self.biggan(z_tensor, class_vector, self.truncation)
            # images are in range [-1, 1], shape (1, 3, H, W)
            images = (images.clamp(-1, 1) + 1) / 2.0  # to [0,1]
            pil_images = convert_to_images(images.cpu())  # returns list of PIL images
            return pil_images[0]

    def encode_text(self, text: str):
        inputs = self.clip_processor(text=[text], images=None, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model.get_text_features(**inputs)
        # normalize
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs

    def encode_image(self, pil_image: Image.Image):
        inputs = self.clip_processor(text=None, images=pil_image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs

    def optimize_latent_for_prompt(self,
                                   prompt: str,
                                   class_name: Optional[str] = None,
                                   steps: int = 300,
                                   lr: float = 0.05,
                                   save_intermediate: bool = True,
                                   intermediate_every: int = 50,
                                   output_dir: str = "outputs"):
        
        os.makedirs(output_dir, exist_ok=True)
        sanitized = sanitize_filename(prompt)
        final_img_path = os.path.join(output_dir, f"{sanitized}_final.png")
        loss_plot_path = os.path.join(output_dir, f"loss_plot_{sanitized}.png")

        # Prepare text features
        text_feat = self.encode_text(prompt)  # shape (1, D)
        text_feat = text_feat.detach()

        # Class vector: if provided, create one-hot via BigGAN util
        if class_name:
            print(f"[INFO] Creating class vector for: {class_name}")
            one_hot = one_hot_from_names([class_name], batch_size=1)
            class_vector = torch.from_numpy(one_hot).float().to(self.device)
        else:
            class_vector = torch.zeros((1, 1000), device=self.device)

        # Initialize latent z as truncated normal sample (consistent with BigGAN sampling)
        z_np = truncated_noise_sample(truncation=self.truncation, batch_size=1, dim_z=self.dim_z)
        z = torch.from_numpy(z_np).float().to(self.device)  # shape (1,128)
        z.requires_grad = True

        optimizer = optim.Adam([z], lr=lr)

        loss_history = []

        pbar = tqdm(range(steps), desc=f"Optimizing '{prompt[:30]}...'")
        for step in pbar:
            optimizer.zero_grad()

            # generate image from current z
            with torch.no_grad():
                # But we need image differentiable w.r.t z; BigGAN in pytorch_pretrained_biggan is differentiable if we DON'T wrap with no_grad.
                pass

            # forward pass (with grad)
            images = self.biggan(z, class_vector, self.truncation)  # shape (1,3,H,W)
            # to 0..1 for CLIP
            images_01 = (images.clamp(-1, 1) + 1) / 2.0
            # convert to PIL-like tensors expected by CLIP processor: the processor accepts PIL Image or numpy image or torch tensor in [0,1]
            # We'll convert to PIL via torchvision or get numpy
            img_np = images_01.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()  # H x W x 3 in [0,1]
            # convert to PIL
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))

            # encode image via CLIP
            image_feat = self.encode_image(pil_img)

            # compute loss: negative cosine similarity (we maximize similarity)
            # similarity = (image_feat * text_feat).sum()
            similarity = F.cosine_similarity(image_feat, text_feat, dim=-1)  # shape (1,)
            loss = -similarity.mean()

            # backprop
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "sim": f"{similarity.item():.4f}"})

            # save intermediate images
            if save_intermediate and (step % intermediate_every == 0 or step == steps - 1):
                # regenerate image (detached) for saving
                with torch.no_grad():
                    out_img = self.generate_biggan_image_from_z(z.detach(), class_vector)
                    inter_path = os.path.join(output_dir, f"{sanitized}_{step}.png")
                    out_img.save(inter_path)

        # Save final image
        final_img = self.generate_biggan_image_from_z(z.detach(), class_vector)
        final_img.save(final_img_path)

        # Save loss plot
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("CLIP Loss (negative similarity)")
        plt.title(f"Loss curve for: {prompt}")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

        return final_img_path, loss_plot_path, loss_history


def load_prompts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = CLIPGuidedBigGAN(device=device,
                                biggan_model_name=args.biggan_model,
                                clip_model_name=args.clip_model,
                                truncation=args.truncation)

    prompts = load_prompts_from_file(args.prompts)
    print(f"[INFO] Loaded {len(prompts)} prompts.")

    for prompt in prompts:
        print(f"\n[INFO] Generating for prompt: {prompt}")
        fp, lp, hist = pipeline.optimize_latent_for_prompt(
            prompt=prompt,
            class_name=None if args.class_name is None else args.class_name,
            steps=args.steps,
            lr=args.lr,
            save_intermediate=args.save_intermediate,
            intermediate_every=args.intermediate_every,
            output_dir=args.output_dir
        )
        print(f"[RESULT] Final image saved to: {fp}")
        print(f"[RESULT] Loss plot saved to: {lp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP + BigGAN latent optimization pipeline")
    parser.add_argument("--prompts", type=str, default="prompts.txt", help="Path to prompts.txt")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--steps", type=int, default=400, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate for z optimizer")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate images")
    parser.add_argument("--intermediate_every", type=int, default=50, help="Save intermediate every N steps")
    parser.add_argument("--biggan_model", type=str, default="biggan-deep-256", help="BigGAN model name")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="HuggingFace CLIP model")
    parser.add_argument("--truncation", type=float, default=0.4, help="BigGAN truncation")
    parser.add_argument("--class_name", type=str, default=None, help="Optional ImageNet class name to condition BigGAN on (e.g. 'golden retriever')")
    args = parser.parse_args()

    main(args)

import argparse
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from FastGAN.update_image import load_model
from FastGAN.lpips import PerceptualLoss


DEVICE = torch.device("cpu")
LATENT_DIM = 256

percept = PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE == "cuda")


def invert(model, target_image, n_images=1, n_iter=100):
    model.eval()
    target_image = (
        torch.Tensor(target_image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .expand(n_images, -1, -1, -1)
        .to(DEVICE)
    )
    z = torch.randn(n_images, LATENT_DIM).to(DEVICE).requires_grad_(True)

    optimizer = optim.Adam([z], lr=0.01)

    for _ in tqdm(range(n_iter), desc="inverting"):
        gen_im = model(z)[0]
        loss = percept(gen_im, target_image)

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    gen_im = model(z)[0].add(1).mul(0.5)
    gen_im = (
        gen_im.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return gen_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate images")
    parser.add_argument("--ckpt", type=str, help="Path to the generator checkpoint.")
    parser.add_argument("-i", dest="input", type=str, help="Input image path.")
    parser.add_argument("-o", dest="output", type=str, help="Output image path.")
    parser.add_argument(
        "--n-iter", type=int, default=100, help="Number of fitting iterations."
    )
    parser.add_argument(
        "--n-image", type=int, default=1, help="Number of images to generate."
    )
    args = parser.parse_args()

    model = load_model(args.ckpt)
    target_im = np.array(Image.open(args.input).convert("RGB"))
    inverted_im = invert(model, target_im, n_iter=args.n_iter, n_images=args.n_image)
    for i in range(inverted_im.shape[0]):
        Image.fromarray(inverted_im[i]).save(args.output.replace(".jpg", f"_{i}.jpg"))

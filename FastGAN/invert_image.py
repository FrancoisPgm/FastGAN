import argparse
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from FastGAN.lpips import PerceptualLoss
from FastGAN.conf import DEVICE

device = torch.device(DEVICE)

# percept = PerceptualLoss(model="net-lin", net="vgg", use_gpu=DEVICE == "cuda")
percept = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)


def invert(model, target_image, n_images=1, n_iter=100, print_best=False):
    model.eval()
    target_image = torch.Tensor(target_image).mul(2.0 / 255).add(-1).permute(2, 0, 1)
    if len(target_image.shape) == 3:
        target_image = target_image.unsqueeze(0).expand(n_images, -1, -1, -1)
    else:
        n_images = target_image.shape[0]
    target_image = target_image.to(device)
    z = torch.Tensor(n_images, 256).normal_(0, 0.6).to(device).requires_grad_(True)

    optimizer = optim.Adam([z], lr=0.01)

    # Progressive lr for FastGAN.lpips
    # optimizer1 = optim.Adam([z], lr=0.1)
    # optimizer2 = optim.Adam([z], lr=0.05)
    # optimizer3 = optim.Adam([z], lr=0.01)

    for _ in tqdm(range(n_iter), desc="inverting"):
        gen_im = model(z)[0].clamp_(-1, 1)
        loss = percept(gen_im, target_image)

        # if i < n_iter // 4:
        #     optimizer = optimizer1
        # elif i < n_iter // 2:
        #     optimizer = optimizer2
        # else:
        #     optimizer = optimizer3

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    gen_im = model(z)[0].clamp_(-1, 1)
    if print_best:
        best_i = 0
        best_loss = percept(gen_im[0:1], target_image[0:1])
        for i in range(1, n_images):
            loss = percept(gen_im[i : i + 1], target_image[0:1])
            if loss < best_loss:
                best_loss = loss
                best_i = i
        print("best:", best_i)
    gen_im = (
        gen_im.add(1)
        .mul(255.0 / 2)
        .add_(0.5)
        .clamp_(0, 255)
        .detach()
        .permute(0, 2, 3, 1)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return gen_im, z.detach().numpy()


if __name__ == "__main__":
    from FastGAN.update_image import load_model

    parser = argparse.ArgumentParser(description="generate images")
    parser.add_argument("--ckpt", type=str, help="Path to the generator checkpoint.")
    parser.add_argument("-i", dest="input", type=str, help="Input image path.")
    parser.add_argument("-o", dest="output", type=str, help="Output image path.")
    parser.add_argument(
        "--n-iter", type=int, default=20, help="Number of fitting iterations."
    )
    parser.add_argument(
        "--n-image", type=int, default=1, help="Number of images to generate."
    )
    args = parser.parse_args()

    model = load_model(args.ckpt)
    target_im = np.array(Image.open(args.input).convert("RGB"))
    inverted_im = invert(
        model,
        target_im,
        n_iter=args.n_iter,
        n_images=args.n_image,
        print_best=args.n_image > 1,
    )[0]
    if args.n_image > 1:
        for i in range(inverted_im.shape[0]):
            Image.fromarray(inverted_im[i]).save(
                args.output.replace(".jpg", f"_{i}.jpg")
            )
    else:
        Image.fromarray(inverted_im[0]).save(args.output)

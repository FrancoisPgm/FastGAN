import torch
import torch.nn.functional as F
from torchvision import utils as vutils

import os
import argparse
from tqdm import tqdm

from FastGAN.models import Generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate images")
    parser.add_argument("--ckpt", type=str, help="Checkpoint weight path.")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Torch device to use."
    )
    parser.add_argument("--batch", default=16, type=int, help="batch size")
    parser.add_argument("--n_sample", type=int, default=200)
    parser.add_argument("--im_size", type=int, default=1024)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    noise_dim = 256
    device = torch.device(args.device)

    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)
    net_ig.to(device)

    checkpoint = torch.load(args.ckpt, map_location=lambda a, b: a)
    net_ig.load_state_dict(checkpoint["g"])

    net_ig.eval()
    net_ig.to(device)

    del checkpoint

    os.makedirs(args.output_path, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(args.n_sample // args.batch)):
            noise = torch.randn(args.batch, noise_dim).to(device)
            g_imgs = net_ig(noise)[0]
            for j, g_img in enumerate(g_imgs):
                vutils.save_image(
                    g_img.add(1).mul(0.5),
                    os.path.join(args.output_path, "%d.png" % (i * args.batch + j)),
                )

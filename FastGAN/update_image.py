import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import utils as vutils
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from FastGAN.models import Generator


SAM2_CKPT = "/Users/fpaugam/Documents/code/capsule_fastgan/sam2/checkpoints/sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEVICE = "cpu"
IM_SIZE = 512

device = torch.device(DEVICE)
sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
predictor = SAM2ImagePredictor(sam2_model)


def load_model(ckpt):
    net_ig = Generator(ngf=64, nz=256, im_size=IM_SIZE)
    net_ig.to(device)
    checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
    net_ig.load_state_dict(checkpoint["g"])
    net_ig.eval()
    net_ig.to(device)
    return net_ig

def gen_image(model, seed):
    gen_im = model(seed.to(device))[0][0].add(1).mul(0.5)
    gen_im = (
        gen_im.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .detach()
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return gen_im


def segment_image(image, point):
    im_array = np.array(image)
    predictor.set_image(im_array)

    masks, _, _ = predictor.predict(
        point_coords=point,
        point_labels=np.array([1]),
        multimask_output=False,
    )

    return masks[0]


def get_bounding_box(mask):
    coords = np.argwhere(mask == 1)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = (x_max - x_min) + 1
    height = (y_max - y_min) + 1
    return x_min, y_min, width, height


def paste_patch(image, gen_im, mask):
    x, y, w, h = get_bounding_box(mask)  # maybe not redo that each time if too slow
    box_size = max(w, h)
    if box_size > IM_SIZE:
        gen_im = cv2.resize(gen_im, dsize=(box_size, box_size))
    else:
        gen_im = gen_im[:, :box_size, :box_size]

    image[mask == 1] = gen_im[:w, :h][mask[x : x + w, y : y + h] == 1]
    return image


def add_new_patch(model, image):
    seed = torch.randn(1, 256)
    point = np.array(
        [[np.random.randint(image.shape[0]), np.random.randint(image.shape[1])]]
    )
    mask = segment_image(image, point)
    if mask.sum() < 5:
        return image, None, seed

    gen_im = gen_image(model, seed)

    return paste_patch(image, gen_im, mask), mask, seed


def update_patch(model, image, mask, seed, scale=0.01):
    seed += scale * torch.randn(1, 256)
    gen_im = gen_image(model, seed)
    return paste_patch(image, gen_im, mask), seed

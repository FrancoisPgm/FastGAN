import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from FastGAN.conf import DEVICE, IM_SIZE, SAM2_CFG, SAM2_CKPT, BOX_SIZE
from FastGAN.models import Generator

device = torch.device(DEVICE)

# The SAM2 predictor is only needed by the segmentation/display process.
# It used to be built at import time, which meant every process that
# imported this module (including the two GAN worker processes, which
# never call segment_image) paid for a redundant SAM2 model on the GPU.
# Loading it lazily means only the process that actually calls
# segment_image() ever instantiates it.
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
        _predictor = SAM2ImagePredictor(sam2_model)
    return _predictor


def load_model(ckpt):
    net_ig = Generator(ngf=64, nz=256, im_size=IM_SIZE)
    net_ig.to(device)
    checkpoint = torch.load(ckpt, map_location=lambda a, b: a)
    net_ig.load_state_dict(checkpoint["g"])
    net_ig.eval()
    # We only ever optimize the latent `z`, never the generator's own
    # weights, so freezing them avoids autograd needlessly computing and
    # storing gradients for every conv layer on each inversion step.
    for p in net_ig.parameters():
        p.requires_grad_(False)
    net_ig.to(device)
    return net_ig


def gen_image(model, seed):
    with torch.no_grad():
        gen_im = model(seed.to(device))[0].add(1).mul(0.5)
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


def segment_image(image, points):
    predictor = get_predictor()
    im_array = np.array(image)
    predictor.set_image(im_array)
    boxes = np.array([
        [
            max(0, p[0] - BOX_SIZE//2),
            max(0, p[1] - BOX_SIZE//2),
            min(p[0] + BOX_SIZE//2, image.shape[0]),
            min(p[1] + BOX_SIZE//2, image.shape[1])
        ] for p in points
    ])[0]

    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=np.array([1] * len(points)),
        multimask_output=False,
        box = boxes
    )

    return masks


def get_bounding_box(mask):
    coords = np.argwhere(mask == 1)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    width = (x_max - x_min) + 1
    height = (y_max - y_min) + 1
    return x_min, y_min, width, height


def get_image_chunks(image, masks):
    """Get the square chunks and bounding boxes for each mask resized to IM_SIZE."""
    boxes = []
    chunks = []
    for mask in masks:
        x, y, w, h = get_bounding_box(mask)
        chunks.append(cv2.resize(image[x : x + w, y : y + h], dsize=(IM_SIZE, IM_SIZE)))
        boxes.append((x, y, w, h))
    return chunks, boxes


def paste_patch(image, gen_im, mask):
    x, y, w, h = get_bounding_box(mask)  # maybe not redo that each time if too slow
    box_size = max(w, h)
    if box_size > IM_SIZE:
        gen_im = cv2.resize(gen_im, dsize=(box_size, box_size))
    else:
        gen_im = gen_im[:, :box_size, :box_size]

    image[mask == 1] = gen_im[:w, :h][mask[x : x + w, y : y + h] == 1]
    return image


def add_new_patch(model, image, image_inversion, n_iter=5):
    """Kept for non-multiprocessing / reference usage. The multiprocessing
    pipeline in mp_pipeline.py reimplements this split across the
    segmentation/display and inversion processes instead of calling it
    directly."""
    from FastGAN.invert_image import invert  # local: only needed here

    seed = torch.randn(1, 256)
    point = np.array(
        [[np.random.randint(image.shape[0]), np.random.randint(image.shape[1])]]
    )
    mask = segment_image(image, point)[0]
    if mask.sum() < 5:
        return image, None, seed

    if image_inversion:
        chunk = get_image_chunks(image, [mask])[0][0]
        gen_im, seed = invert(model, chunk, n_iter=n_iter)
        gen_im = gen_im[0]
    else:
        gen_im = gen_image(model, seed)[0]

    return paste_patch(image, gen_im, mask), mask, seed


def update_patch(model, image, mask, seed, scale=0.01):
    """Kept for non-multiprocessing / reference usage. mp_pipeline.py's
    update_worker reimplements this as a batched update across all active
    patches instead of one at a time."""
    seed += scale * torch.randn(1, 256)
    gen_im = gen_image(model, seed)
    return paste_patch(image, gen_im, mask), seed

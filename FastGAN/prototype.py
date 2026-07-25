import torch
from FastGAN.update_image import (
    add_new_patch,
    paste_patch,
    load_model,
    gen_image,
    device,
)
import time
import numpy as np
from PIL import Image
import cv2

from FastGAN.conf import GAN_CKPT, IM_PATH

im = Image.open(IM_PATH)
im = np.array(im.convert("RGB"))

image_inversion = True

# TODO: optimize with multiprocessings
# maybe one for inverting images and one or two for updating patches

window_name = "proto"
cv2.namedWindow(window_name)

masks = []
all_seeds = None
model = load_model(GAN_CKPT)
model.eval()

mask = None
while mask is None:
    im, mask, all_seeds = add_new_patch(model, im, image_inversion)
masks.append(mask)

display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
cv2.imshow(window_name, display_img)
cv2.waitKey(1)

while True:
    im, mask, seed = add_new_patch(model, im, image_inversion)
    if mask is not None:
        masks.append(mask)
        all_seeds = np.vstack((all_seeds, seed))
    if len(masks) > 30:
        masks.pop(0)
        all_seeds = all_seeds[1:]
    display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)

    for _ in range(15):
        all_seeds += np.random.randn(*all_seeds.shape) * 0.1
        new_gen_im = gen_image(model, torch.Tensor(all_seeds).to(device))
        for i in range(len(masks)):
            im = paste_patch(im, new_gen_im[i], masks[i])
        display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)

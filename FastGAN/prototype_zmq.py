import time

import numpy as np
import torch
import zmq

from FastGAN.conf import GAN_CKPT, IM_PATH, N_ITER
from FastGAN.update_image import (
    add_new_patch,
    device,
    gen_image,
    load_model,
    paste_patch,
)

WIDTH = 682
HEIGHT = 512
CHANNELS = 4


image_inversion = False

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

print(f"Server started... Expecting {WIDTH}x{HEIGHT} images")

masks = []
all_seeds = []
model = load_model(GAN_CKPT)
model.eval()

print("waiting for first blast")
message = socket.recv()
print("First blast received")
im = np.frombuffer(message, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
alpha = np.zeros(im.shape[:2] + (1,), dtype=np.uint8)

while True:
    # 1. Check for a "Blast" (New initial image)
    try:
        # Use NONBLOCK so the server doesn't stop processing the current loop
        message = socket.recv(flags=zmq.NOBLOCK)
        im = (
            np.frombuffer(message, dtype=np.uint8)
            .reshape(HEIGHT, WIDTH, CHANNELS)
            .copy()
        )
        alpha = np.zeros(im.shape[:2] + (1,), dtype=np.uint8)
        print("Blast received! Resetting loop.")
    except zmq.Again:
        pass  # No new blast, keep processing
    im = im[::-1, :, :3]

    im, mask, seed = add_new_patch(model, im, image_inversion, n_iter=N_ITER)
    # alpha[:, :, 0] = np.clip((alpha[:, :, 0] + mask * 255).astype(np.uint8), 0, 255)
    # alpha[:150, :150] = 255
    if mask is not None:
        masks.append(mask)
        all_seeds.append(seed)
    if len(masks) > 30:
        masks.pop(0)
        all_seeds = all_seeds[1:]

    im = np.concatenate([im[::-1], alpha], axis=2)
    im[:, :, 0] *= np.clip(sum(masks), 0, 1).astype(np.uint8)
    im[:, :, 1] *= np.clip(sum(masks), 0, 1).astype(np.uint8)
    im[:, :, 2] *= np.clip(sum(masks), 0, 1).astype(np.uint8)

    socket.send(im.tobytes())

    # # update patches
    # for _ in range(15):
    #     all_seeds += np.random.randn(*all_seeds.shape) * 0.1
    #     new_gen_im = gen_image(model, torch.Tensor(all_seeds).to(device))
    #     for i in range(len(masks)):
    #         im = paste_patch(im, new_gen_im[i], masks[i])

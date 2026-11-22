import os
import pickle
import time

import torch
import zmq
from PIL import Image

from FastGAN.conf import BATCH_SIZE, DIAPO_DIR, DIFF_STEP, IMAGES_MAX, MODELS_DIAPO
from FastGAN.update_image import gen_image, load_model


def make_batch_seeds(base_seed):
    """Add cumulative noise to the base seed to make the batch's seeds."""
    noise = torch.randn(BATCH_SIZE, 256) * DIFF_STEP
    noise[0] += base_seed
    new_seeds = torch.cumsum(noise, dim=0)
    return new_seeds


context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5556")

print("Server started...")

models = {}
for biome, path in MODELS_DIAPO.items():
    models[biome] = load_model(path)

for m in models.values():
    m.eval()


print("Models loaded, waiting for first request.")

message = pickle.loads(socket.recv())
biome = message["biome"]
model = models[biome]
seed = torch.randn(256)
print(f"First request received, biome {biome}.")

i = 1
created_images_paths = []

while True:
    try:
        message = socket.recv(flags=zmq.NOBLOCK)
        message = pickle.loads(message)
        biome = message["biome"]
        model = models[biome]
        seed = torch.randn(256)
        i = 1
        for p in created_images_paths:
            os.remove(p)
        created_images_paths = []
        print(f"New request received, biome {biome}.")
    except zmq.Again:
        pass

    batch_seeds = make_batch_seeds(seed)
    seed = batch_seeds[-1]
    new_images = gen_image(model, batch_seeds)

    for k in range(BATCH_SIZE):
        path = os.path.join(DIAPO_DIR, f"{biome}_generation_{i + k:05d}.png")
        Image.fromarray(new_images[k]).save(path)
        created_images_paths.append(path)

    if len(created_images_paths) > IMAGES_MAX:
        for _ in range(len(created_images_paths) - IMAGES_MAX):
            p = created_images_paths.pop(0)
            os.remove(p)

    i += BATCH_SIZE
    time.sleep(message["sleep"])

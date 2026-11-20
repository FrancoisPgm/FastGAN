import time
import numpy as np
import pickle
import zmq

from FastGAN.update_image import (
    add_patch_from_class,
    load_model,
    load_classifier,
)
from FastGAN.conf import MODELS_SELECT_REPLACE, CLASSIFIERS

keep_patches = True
# keep_every = 10

context = zmq.Context()
socket = context.socket(zmq.PAIR)
# socket.setsockopt(zmq.SNDHWM, 1)
# socket.setsockopt(zmq.RCVHWM, 1)
socket.bind("tcp://*:5555")

print("Server started...")

models = {}
for biome in MODELS_SELECT_REPLACE:
    models[biome] = {
        key: load_model(value) for key, value in MODELS_SELECT_REPLACE[biome].items()
    }

for d in models.values():
    for m in d.values():
        m.eval()

classifiers = {biome: load_classifier(CLASSIFIERS[biome]) for biome in CLASSIFIERS}

is_blast = False


print("waiting for first blast")

while not is_blast:
    message = pickle.loads(socket.recv())
    is_blast = message["is_blast"]
print("First blast received")

im_init = message["image"]
im = im_init[:, :, :3].copy()
alpha = im_init[:, :, 3:]
total_mask = np.zeros(im_init.shape[:2])

i = 1

while True:
    # 1. Check for a "Blast" (New initial image) and DRAIN the buffer
    new_message = None
    try:
        # This loop runs as fast as possible to clear the ZMQ buffer
        while True:
            try:
                # We use NOBLOCK to check if there is anything else waiting
                msg = socket.recv(flags=zmq.NOBLOCK)
                new_message = pickle.loads(msg)
            except zmq.Again:
                # No more messages in the ZMQ buffer
                break
    except Exception as e:
        print(f"Error receiving: {e}")

    # Now, only process if we actually received something
    if new_message is not None:
        message = new_message  # Use the latest message found

        # Check for "Blast" logic
        if message["is_blast"]:
            im_init = message["image"]
            im = im_init[:, :, :3].copy()
            alpha = im_init[:, :, 3:]
            biome = message["biome"]
            i = 1
            total_mask = np.zeros(im_init.shape[:2])
            print("Blast received! Resetting loop.")

    if not keep_patches:
        im = im_init[:, :, :3].copy()
        alpha = im_init[:, :, 3:]

    classifier, labels = classifiers[biome]
    im, mask, seed, box = add_patch_from_class(models[biome], im, classifier, labels)
    total_mask[mask == 1] = 1

    if not keep_patches:
        total_mask = mask

    box_im = np.zeros((*im.shape[:2], 4), dtype=np.uint8)
    rows, cols = np.where(mask == 1)
    x, y, w, h = box
    box_im[y, x : x + w - 1] = [255, 255, 255, 255]
    box_im[y + h - 1, x : x + w - 1] = [255, 255, 255, 255]
    box_im[y : y + h - 1, x] = [255, 255, 255, 255]
    box_im[y : y + h - 1, x + w - 1] = [255, 255, 255, 255]

    im_to_send = np.concatenate([im, alpha], axis=2).copy()
    im_to_send[total_mask == 0, :] = 0

    socket.send(pickle.dumps({"image": im_to_send, "box": box_im}))
    i += 1
    time.sleep(message["sleep"])

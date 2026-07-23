from FastGAN.update_image import add_new_patch, update_patch, load_model
import time
import numpy as np
from PIL import Image
import cv2

GAN_CKPT = "/Users/fpaugam/Documents/code/capsule_fastgan/output/test_capsule/train_results/test_512_cpu_3/models/30000.pth"
im = Image.open("../../Images_eve_jpeg/IMG_5448.jpeg")
im = np.array(im.convert("RGB"))

# TODO: optimize with multiprocessing
# IMG_SIZE = 1024
# IMG_CHANNELS = 3
# BUFFER_SIZE = 50

# # Total size needed for the image data buffer
# BYTES_PER_IMAGE = IMG_SIZE * IMG_SIZE * IMG_CHANNELS
# TOTAL_BUFFER_SIZE = BYTES_PER_IMAGE * BUFFER_SIZE

window_name = "proto"
cv2.namedWindow(window_name)

patches = []
model = load_model(GAN_CKPT)

while True:
    im, mask, seed = add_new_patch(model, im)
    if mask is not None:
        patches.append((mask, seed))
    if len(patches) > 100:
        patches.pop(0)
    display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(1)

    for _ in range(15):
        for i in range(len(patches)):
            mask, seed = patches[i]
            im, new_seed = update_patch(model, im, mask, seed)
            patches[i] = (mask, new_seed)
        display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)


# def generate_images(shm_name):
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     shared_images = np.ndarray(
#         (BUFFER_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
#         dtype=np.uint8,
#         buffer=existing_shm.buf
#     )
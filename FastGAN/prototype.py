from FastGAN.update_image import add_new_patch, update_patch
import time
import numpy as np
from PIL import Image
import cv2

im = Image.open("../../Images_eve_jpeg/IMG_5448.jpeg")
im = np.array(im.convert("RGB"))

cv2.namedWindow("test")

patches = []

while True:
    im, mask, seed = add_new_patch(im)
    if mask is not None:
        patches.append((mask, seed))
    if len(patches) > 100:
        patches.pop(0)
    display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow("test", display_img)
    cv2.waitKey(1)

    for _ in range(15):
        for i in range(len(patches)):
            mask, seed = patches[i]
            im, new_seed = update_patch(im, mask, seed)
            patches[i] = (mask, new_seed)
        display_img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow("test", display_img)
        cv2.waitKey(1)

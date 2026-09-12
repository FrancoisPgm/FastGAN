import numpy as np
import pickle
import zmq

from FastGAN.update_image import add_patch_from_class, load_model


keep_patches = True
# keep_every = 10

context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("tcp://*:5555")

print("Server started...")

models = []
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/rocher.pth")
# )
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/arbre.pth")
# )
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/sol.pth")
# )
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/sol.pth")
# )
# models.append(
#     load_model(
#         "/Users/fpaugam/Documents/code/capsule_fastgan/test_models/feuillage.pth"
#     )
# )
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/ciel.pth")
# )
# models.append(
#     load_model("/Users/fpaugam/Documents/code/capsule_fastgan/test_models/metal.pth")
# )


models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
models.append(load_model("/Users/fpaugam/Downloads/all_50000.pth"))
for m in models:
    m.eval()

print("waiting for first blast")
message = pickle.loads(socket.recv())
print("First blast received")
# im_init = np.frombuffer(message, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS).copy()
im_init = message["image"]
im = im_init[:, :, :3].copy()
alpha = im_init[:, :, 3:]
total_mask = np.zeros(im_init.shape[:2])

i = 1

while True:
    # 1. Check for a "Blast" (New initial image)
    try:
        # Use NONBLOCK so the server doesn't stop processing the current loop
        message = pickle.loads(socket.recv(flags=zmq.NOBLOCK))
        # im_init = (
        #     np.frombuffer(message, dtype=np.uint8)
        #     .reshape(HEIGHT, WIDTH, CHANNELS)
        #     .copy()
        # )
        im_init = message["image"]
        im = im_init[:, :, :3].copy()
        alpha = im_init[:, :, 3:]
        if message["is_blast"]:
            i = 1
            total_mask = np.zeros(im_init.shape[:2])
            print("Blast received! Resetting loop.")
        else:
            print("image received")
    except zmq.Again:
        pass  # No image received, keep processing

    if not keep_patches:
        im = im_init[:, :, :3].copy()
        alpha = im_init[:, :, 3:]

    im, mask, seed, box = add_patch_from_class(models, im)
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

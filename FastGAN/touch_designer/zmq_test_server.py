import zmq
import numpy as np
import cv2


def process_image(rgba_img):
    # 1. Split the image into RGB and Alpha
    # rgba_img shape is (720, 1280, 4)
    rgb = rgba_img[:, :, :3].copy()  # First 3 channels (RGB)
    alpha = rgba_img[:, :, 3:]  # The 4th channel (Alpha)

    # --- YOUR IMAGE PROCESSING LOGIC HERE ---
    # Note: OpenCV usually expects BGR. TD sends RGB.
    # If you use cv2.cvtColor, remember it's RGB -> GRAY here.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Example: Create a mask where pixels are brighter than 128
    mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

    # Example: Turn the masked area Green
    rgb[mask > 0] = [0, np.random.randint(12, 255), np.random.randint(12, 255)]
    # ----------------------------------------

    # 2. Combine the processed RGB and the original Alpha back together
    # This ensures that transparency is preserved when it goes back to TD
    result = np.concatenate([rgb, alpha], axis=2)

    return result


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# Update these to match your actual Movie File In resolution
WIDTH = 1280
HEIGHT = 720
CHANNELS = 4

print(f"Server started... Expecting {WIDTH}x{HEIGHT} RGBA images")

while True:
    # Receive bytes
    message = socket.recv()
    print("received")

    # Convert bytes to numpy array with 4 channels
    img = np.frombuffer(message, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)

    # Process the RGBA image
    processed_img = process_image(img)

    # Send the 4-channel image back as bytes
    socket.send(processed_img.tobytes())
    print("sent")

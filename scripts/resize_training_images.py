import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm


def resize_images(input_folder, output_folder, target_dim=2048):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_path = Path(input_folder)
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

    for img_path in tqdm(image_files):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Calculate scale factor
            scale = target_dim / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)

            # INTER_AREA is the preferred interpolation for shrinking images
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            save_path = os.path.join(output_folder, img_path.name)
            cv2.imwrite(save_path, resized_img)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize training images")
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str)
    parser.add_argument("--size", type=int, default=2048, help="Largest dimension.")
    args = parser.parse_args()

    resize_images(args.input_dir, args.output_dir, args.size)

#!/usr/bin/env python3
"""
a4_grid_montage_one_dim_overflow_corrected.py

- 2x2 A4 montage
- Image fills the cell as much as possible
- Faces + main subject stay fully inside
- Only one dimension may overflow
- Minimal cropping, full-image scaling
- Margins, rotation, multi-page intact
"""

import os
import glob
from PIL import Image
import cv2
import numpy as np

# ---------- Config ----------
DPI = 300
A4_PX = (2480, 3508)
ROWS = 2
COLS = 2
GRID_MARGIN = 40
MARGIN_OUTER = 60
INNER_CELL_MARGIN = 20
MAX_PER_PAGE = ROWS * COLS
OUTPUT_DIR = "montage_pages"
PREVIEW_AUTOSHOW = True
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# ----------------------------

TOTAL_H_MARGIN = (COLS - 1) * GRID_MARGIN + 2 * MARGIN_OUTER + 2 * INNER_CELL_MARGIN * COLS
TOTAL_V_MARGIN = (ROWS - 1) * GRID_MARGIN + 2 * MARGIN_OUTER + 2 * INNER_CELL_MARGIN * ROWS
CELL_W = (A4_PX[0] - TOTAL_H_MARGIN) // COLS
CELL_H = (A4_PX[1] - TOTAL_V_MARGIN) // ROWS


def find_images_in_cwd():
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(ext))
    files = sorted(list({f.lower(): f for f in files}.values()))
    return files


def is_landscape(pil_img: Image.Image):
    return pil_img.width > pil_img.height


def detect_faces_bbox(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        h, w = gray.shape
        return int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)
    x1 = min([x for (x,y,w,h) in faces])
    y1 = min([y for (x,y,w,h) in faces])
    x2 = max([x+w for (x,y,w,h) in faces])
    y2 = max([y+h for (x,y,w,h) in faces])
    return x1, y1, x2-x1, y2-y1


def detect_subject_bbox(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def place_image_in_cell(pil_img: Image.Image, cell_w, cell_h):
    if is_landscape(pil_img):
        pil_img = pil_img.rotate(90, expand=True)

    img_w, img_h = pil_img.size
    np_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Detect faces and subject
    fx, fy, fw, fh = detect_faces_bbox(np_img)
    sx, sy, sw, sh = detect_subject_bbox(np_img)

    # Combined bbox
    x1 = min(fx, sx)
    y1 = min(fy, sy)
    x2 = max(fx+fw, sx+sw)
    y2 = max(fy+fh, sy+sh)
    bbox_cx = (x1 + x2)/2
    bbox_cy = (y1 + y2)/2
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Scale the full image
    scale_x = cell_w / img_w
    scale_y = cell_h / img_h
    # Choose scale so that only one dimension can overflow
    if scale_x < scale_y:
        scale = scale_y  # width may overflow
    else:
        scale = scale_x  # height may overflow

    new_w = int(round(img_w * scale))
    new_h = int(round(img_h * scale))
    pil_resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

    # scale combined bbox
    bbox_cx *= scale
    bbox_cy *= scale

    # Shift image so combined bbox fully inside cell
    max_crop_x = max(new_w - cell_w, 0)
    max_crop_y = max(new_h - cell_h, 0)
    crop_x = int(np.clip(bbox_cx - cell_w/2, 0, max_crop_x))
    crop_y = int(np.clip(bbox_cy - cell_h/2, 0, max_crop_y))

    return pil_resized.crop((crop_x, crop_y, crop_x+cell_w, crop_y+cell_h))


def make_pages(image_files):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pages = []
    for pidx in range(0, len(image_files), MAX_PER_PAGE):
        batch = image_files[pidx:pidx+MAX_PER_PAGE]
        canvas = Image.new("RGB", A4_PX, (255,255,255))
        for i, img_path in enumerate(batch):
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: failed to open {img_path}: {e}")
                continue
            cell_img = place_image_in_cell(pil_img, CELL_W, CELL_H)
            row = i // COLS
            col = i % COLS
            x = MARGIN_OUTER + col*(CELL_W+GRID_MARGIN+2*INNER_CELL_MARGIN)+INNER_CELL_MARGIN
            y = MARGIN_OUTER + row*(CELL_H+GRID_MARGIN+2*INNER_CELL_MARGIN)+INNER_CELL_MARGIN
            canvas.paste(cell_img, (int(x), int(y)))
        out_name = os.path.join(OUTPUT_DIR, f"page_{(pidx//MAX_PER_PAGE)+1:03d}.png")
        canvas.save(out_name, dpi=(DPI, DPI))
        pages.append(out_name)
        print(f"Saved: {out_name}")
        if PREVIEW_AUTOSHOW:
            try:
                Image.open(out_name).show()
            except Exception as e:
                print(f"Could not preview {out_name}: {e}")
    return pages


def main():
    images = find_images_in_cwd()
    if not images:
        print("No images found in current directory.")
        return
    print(f"Found {len(images)} images.")
    pages = make_pages(images)
    print("Done. Pages created:")
    for p in pages:
        print("  ", p)


if __name__ == "__main__":
    main()

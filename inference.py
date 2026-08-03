"""
inference.py

Real inference pipeline for ai_pacs.

This is a refactor of the original exploratory `test.py` notebook
script: same models, same steps, same logic -- but restructured to run
on a single in-memory DICOM dataset (as received by listener.py) instead
of batch-reading a list of file paths from an Excel sheet.

Pipeline per image:
    1. Classify eye side (Left/Right) with a Swin binary classifier.
    2. Locate the optic disc with a YOLO detector (confidence backs
       off automatically in run_yolo_dynamic_conf() until *something*
       is found).
    3. Crop a fixed-size window around the detection, offset by eye
       side (calculate_crop_bounds()).
    4. Run a segmentation model (StrongModel / Unet w/ mit_b4 encoder)
       on the crop to get a binary mask.
    5. Paste the crop-space mask back into a full-resolution canvas
       matching the source image, so it lines up pixel-for-pixel with
       the DICOM instance listener.py received. This is what makes it
       usable as a DICOM-SEG, which must reference the source image.

listener.py calls run_inference(ds) and gets back an InferenceResult:
the full-resolution mask (fed into build_segmentation() to make the
DICOM-SEG) plus a blended overlay screenshot (for QA/logging) and some
auxiliary info (eye side, crop box, detection confidence).

MODEL SETUP: model checkpoints and the `models` module (StrongModel,
SwinBinaryClassifier) are assumed to already be present in the
deployment environment -- they are NOT included in this repo. Place
the checkpoint files alongside listener.py (or set the AI_PACS_*_PATH
environment variables below to point elsewhere) and provide a
models.py exposing StrongModel and SwinBinaryClassifier before running.
"""

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pydicom
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from models import StrongModel, SwinBinaryClassifier

logger = logging.getLogger("ai_pacs.inference")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE_YOLO = 512
CROP_SIZE = 512
LABEL_MAP = {0: "Left", 1: "Right"}

# Checkpoint paths -- override via environment variables if the files
# don't live next to this script.
SEGMENTATION_MODEL_PATH = os.environ.get(
    "AI_PACS_SEG_MODEL_PATH",
    "50epochs_strongmodel_augmentation_gencropping_noclahe_8020split_mitUNET.pth",
)
SWIN_MODEL_PATH = os.environ.get("AI_PACS_SWIN_MODEL_PATH", "swintiny_binary_best.pth")
YOLO_MODEL_PATH = os.environ.get("AI_PACS_YOLO_MODEL_PATH", "Yolo1.pt")

SWIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TO_TENSOR = transforms.ToTensor()


@dataclass
class InferenceResult:
    """Everything listener.py needs to build a DICOM-SEG and log/QA the result."""
    mask: np.ndarray          # (H, W) uint8, values 0/1 -- full source-image resolution
    screenshot: np.ndarray    # (crop_h, crop_w, 3) uint8, BGR -- crop with red mask overlay
    eye_label: str            # "Left" or "Right"
    crop_box: tuple           # (x1, y1, x2, y2) in source-image pixel coordinates
    yolo_confidence: float    # confidence threshold that finally produced a detection


class InferenceUnavailableError(RuntimeError):
    """No detection could be produced for this image.

    This is an expected outcome for some images (e.g. poor quality,
    nothing resembling the target anatomy), not a bug -- callers
    should treat it as "skip this instance", not as a processing
    failure.
    """


# --------------------------------------------------------------------------
# Model loading (once, lazily, cached at module scope)
# --------------------------------------------------------------------------

_models = None


def _load_models():
    """Loads all three models once and caches them for the life of the process."""
    global _models
    if _models is not None:
        return _models

    logger.info("Loading ai_pacs inference models onto %s ...", DEVICE)

    seg_model = StrongModel("Unet", "mit_b4", in_channels=3, out_classes=1, encoder_weights=None)
    checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location="cpu")
    seg_model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    seg_model.to(DEVICE).eval()

    swin_model = SwinBinaryClassifier(pretrained=False).to(DEVICE)
    swin_checkpoint = torch.load(SWIN_MODEL_PATH, map_location=DEVICE)
    swin_model.load_state_dict(swin_checkpoint["model_state_dict"])
    swin_model.eval()

    yolo_model = YOLO(YOLO_MODEL_PATH)

    _models = (seg_model, swin_model, yolo_model)
    logger.info("Models loaded.")
    return _models


# --------------------------------------------------------------------------
# Pipeline steps (logic unchanged from test.py, just parameterized)
# --------------------------------------------------------------------------

def dicom_to_pil(ds: pydicom.Dataset) -> Image.Image:
    """Normalizes a DICOM dataset's pixel data to an RGB PIL Image."""
    img = ds.pixel_array.astype(np.float32)

    if img.size == 0:
        raise ValueError("Empty pixel array")

    # 1. Remove extra single-dimensional entries (e.g., shape (1, H, W) -> (H, W))
    img = np.squeeze(img)

    # 2. Invert MONOCHROME1 grayscale if applicable
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    # 3. Min-Max Normalize to [0, 255] uint8
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)
        
    img = img.astype(np.uint8)

    # 4. Handle conversion to 3-channel RGB for PIL
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 1:  # (H, W, 1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[-1] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim == 3 and img.shape[-1] == 3:  # Already RGB (e.g. YBR or RGB DICOM)
        # Handle BGR/YBR DICOMs if needed, or leave as RGB
        pass
    else:
        raise ValueError(f"Unsupported image shape after squeeze: {img.shape}")

    return Image.fromarray(img)


def predict_eye_side(img: Image.Image, model) -> str:
    """Classifies whether the image is a Left or Right eye."""
    input_tensor = SWIN_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return LABEL_MAP[int(prob > 0.5)]


def run_yolo_dynamic_conf(img: Image.Image, model) -> tuple:
    """Runs YOLO inference, backing off confidence until a box is found."""
    confidences = list(np.arange(0.5, 0.1, -0.1))
    confidences += [0.1, 0.08, 0.05, 0.03, 0.01, 0.005, 0.001]

    for conf in confidences:
        conf = round(float(conf), 6)
        try:
            results = model.predict(source=img, imgsz=IMG_SIZE_YOLO, device=DEVICE, conf=conf, verbose=False)
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None and len(boxes) > 0:
                return results, conf
        except Exception:
            logger.exception("YOLO predict error at conf=%s", conf)
    return None, None


def calculate_crop_bounds(cx: int, cy: int, eye_label: str, img_shape: tuple) -> tuple:
    """Calculates a directional CROP_SIZE x CROP_SIZE box around the detection center.

    Assumes the source image is at least CROP_SIZE in both dimensions
    (true for the fundus photos this was built for). Smaller inputs
    will produce a crop narrower than CROP_SIZE.
    """
    h, w = img_shape[:2]
    cx_offset = cx + 50 if eye_label == "Left" else cx - 50

    x1 = max(0, cx_offset - CROP_SIZE // 2)
    y1 = max(0, cy - CROP_SIZE // 2)
    x2 = x1 + CROP_SIZE
    y2 = y1 + CROP_SIZE

    if x2 > w:
        x1, x2 = max(0, w - CROP_SIZE), w
    if y2 > h:
        y1, y2 = max(0, h - CROP_SIZE), h

    return int(x1), int(y1), int(x2), int(y2)


def run_segmentation_model(cropped_img_np: np.ndarray, seg_model) -> np.ndarray:
    """Generates a binary segmentation mask (crop_h x crop_w) from a cropped image."""
    cropped_pil = Image.fromarray(cropped_img_np)
    input_seg = TO_TENSOR(cropped_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask_logit = seg_model(input_seg)
        mask_prob = torch.sigmoid(mask_logit)
        mask_pred = (mask_prob > 0.5).cpu().numpy()[0, 0]

    mask = np.zeros(cropped_img_np.shape[:2], dtype=np.uint8)
    mask[mask_pred > 0] = 1
    return mask


def build_overlay_screenshot(cropped_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    """Blends a red mask over the cropped image (BGR) for QA/visualization."""
    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    colored_mask = np.zeros_like(cropped_bgr, dtype=np.uint8)
    colored_mask[crop_mask > 0] = (0, 0, 255)  # red in BGR
    return cv2.addWeighted(cropped_bgr, 0.7, colored_mask, 0.3, 0)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def run_inference(ds: pydicom.Dataset) -> InferenceResult:
    """Runs the full eye-side + detection + segmentation pipeline on one
    in-memory DICOM dataset.

    Returns an InferenceResult whose `mask` is full source-image
    resolution (0/1, uint8) -- ready to hand to
    listener.build_segmentation() -- plus a `screenshot` overlay for
    QA/logging.

    Raises InferenceUnavailableError if no detection could be made.
    This is an expected "nothing found" outcome for some images, not a
    bug -- callers should skip the instance rather than treat it like
    a processing failure.
    """
    seg_model, swin_model, yolo_model = _load_models()

    img = dicom_to_pil(ds)
    img_np = np.array(img)

    eye_label = predict_eye_side(img, swin_model)
    logger.info("Predicted eye side: %s", eye_label)

    results, final_conf = run_yolo_dynamic_conf(img, yolo_model)
    if not results:
        raise InferenceUnavailableError("YOLO found no detections at any confidence level")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    if len(boxes) == 0:
        raise InferenceUnavailableError("YOLO returned empty boxes")

    best_idx = np.argmax(scores)
    x1, y1, x2, y2 = boxes[best_idx]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    x1_c, y1_c, x2_c, y2_c = calculate_crop_bounds(cx, cy, eye_label, img_np.shape)
    cropped_rgb = img_np[y1_c:y2_c, x1_c:x2_c]

    crop_mask = run_segmentation_model(cropped_rgb, seg_model)

    # Paste the crop-space mask back into a full-resolution canvas so it
    # lines up pixel-for-pixel with the source DICOM image.
    full_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    full_mask[y1_c:y2_c, x1_c:x2_c] = crop_mask

    screenshot = build_overlay_screenshot(cropped_rgb, crop_mask)

    return InferenceResult(
        mask=full_mask,
        screenshot=screenshot,
        eye_label=eye_label,
        crop_box=(x1_c, y1_c, x2_c, y2_c),
        yolo_confidence=final_conf,
    )

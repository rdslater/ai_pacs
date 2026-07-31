import os
import traceback
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from models import StrongModel, SwinBinaryClassifier

# --- Configuration & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./screenshotscropped"
CROP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cropped")
EXCEL_PATH = r"./data.xlsx"

IMG_SIZE_YOLO = 512
CROP_SIZE = 512
LABEL_MAP = {0: "Left", 1: "Right"}

# Model Paths
SEGMENTATION_MODEL_PATH = "50epochs_strongmodel_augmentation_gencropping_noclahe_8020split_mitUNET.pth"
SWIN_MODEL_PATH = "swintiny_binary_best.pth"
YOLO_MODEL_PATH = "Yolo1.pt"

# --- Image Transforms ---
SWIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
TO_TENSOR = transforms.ToTensor()


# --- Helper Functions ---

def dicom_to_pil(dicom_path: str) -> Image.Image:
    """Reads a DICOM file and normalizes it to an RGB PIL Image."""
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    if img.size == 0:
        raise ValueError("Empty pixel array")

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(img)


def load_models():
    """Initializes and loads weights for all 3 models."""
    # 1. Segmentation Model
    seg_model = StrongModel('Unet', 'mit_b4', in_channels=3, out_classes=1, encoder_weights=None)
    checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location="cpu")
    seg_model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    seg_model.to(DEVICE).eval()

    # 2. Swin Classifier
    swin_model = SwinBinaryClassifier(pretrained=False).to(DEVICE)
    swin_checkpoint = torch.load(SWIN_MODEL_PATH, map_location=DEVICE)
    swin_model.load_state_dict(swin_checkpoint['model_state_dict'])
    swin_model.eval()

    # 3. YOLO Object Detector
    yolo_model = YOLO(YOLO_MODEL_PATH)

    return seg_model, swin_model, yolo_model


def predict_eye_side(img: Image.Image, model) -> str:
    """Classifies whether the image is a Left or Right eye."""
    input_tensor = SWIN_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    pred_class = int(prob > 0.5)
    return LABEL_MAP[pred_class]


def run_yolo_dynamic_conf(img: Image.Image, model) -> tuple:
    """Runs YOLO inference dropping confidence iteratively until a box is found."""
    # Strategy 1: Decay from 0.5 down to 0.1
    confidences = list(np.arange(0.5, 0.1, -0.1))
    # Strategy 2: Fine-grained fallback thresholds
    confidences += [0.1, 0.08, 0.05, 0.03, 0.01, 0.005, 0.001]

    for conf in confidences:
        conf = round(float(conf), 6)
        try:
            results = model.predict(source=img, imgsz=IMG_SIZE_YOLO, device=DEVICE, conf=conf, verbose=False)
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None and len(boxes) > 0:
                return results, conf
        except Exception as e:
            print(f"YOLO predict error at conf={conf}: {e}")
    return None, None


def calculate_crop_bounds(cx: int, cy: int, eye_label: str, img_shape: tuple) -> tuple:
    """Calculates directional crop bounding box around optic disc center."""
    h, w, _ = img_shape
    cx_offset = cx + 50 if eye_label == "Left" else cx - 50

    x1 = max(0, cx_offset - CROP_SIZE // 2)
    y1 = max(0, cy - CROP_SIZE // 2)
    x2 = x1 + CROP_SIZE
    y2 = y1 + CROP_SIZE

    # Out of bounds corrections
    if x2 > w:
        x1, x2 = w - CROP_SIZE, w
    if y2 > h:
        y1, y2 = h - CROP_SIZE, h

    return int(x1), int(y1), int(x2), int(y2)


def process_segmentation(cropped_img_np: np.ndarray, seg_model) -> np.ndarray:
    """Generates a binary segmentation mask from a cropped image snippet."""
    cropped_pil = Image.fromarray(cropped_img_np)
    input_seg = TO_TENSOR(cropped_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mask_logit = seg_model(input_seg)
        mask_prob = torch.sigmoid(mask_logit)
        mask_pred = (mask_prob > 0.5).cpu().numpy()[0, 0]
        
    mask_overlay = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    mask_overlay[mask_pred > 0] = 255
    return mask_overlay


def save_overlay_result(cropped_bgr: np.ndarray, mask: np.ndarray, save_path: str):
    """Blends a red mask over the BGR cropped image and saves it to disk."""
    colored_mask = np.zeros_like(cropped_bgr, dtype=np.uint8)
    colored_mask[mask > 0] = (0, 0, 255)  # Red mask in BGR
    blended = cv2.addWeighted(cropped_bgr, 0.7, colored_mask, 0.3, 0)
    cv2.imwrite(save_path, blended)


# --- Main Pipeline ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)

    print("Loading deep learning frameworks...")
    seg_model, swin_model, yolo_model = load_models()
    
    df = pd.read_excel(EXCEL_PATH)
    total_files = len(df)

    for idx, raw_path in enumerate(df["File Path"], start=1):
        dicom_path = str(raw_path).strip()
        base_name = os.path.splitext(os.path.basename(dicom_path))[0]
        print(f"\n[{idx}/{total_files}] Processing {base_name}")

        if not os.path.exists(dicom_path):
            print("→ File missing. Skipping.")
            continue

        try:
            # 1. Load DICOM
            img = dicom_to_pil(dicom_path)
            img_np = np.array(img)

            # 2. Eye Classification (Swin)
            eye_label = predict_eye_side(img, swin_model)
            print(f"→ Classification: {eye_label} Eye")

            # 3. Target Detection (YOLO)
            results, final_conf = run_yolo_dynamic_conf(img, yolo_model)
            if not results:
                print(f"→ No YOLO detections for {base_name}")
                continue
            
            # Save raw YOLO prediction screenshot
            results[0].save(filename=os.path.join(OUTPUT_DIR, f"{base_name}_pred.jpg"))
            
            # Extract highest-scoring detection box
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            if len(boxes) == 0:
                print("→ Found empty bounding boxes. Skipping crop.")
                continue

            best_idx = np.argmax(scores)
            x1, y1, x2, y2 = boxes[best_idx]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 4. Contextual Crop Extraction
            x1_c, y1_c, x2_c, y2_c = calculate_crop_bounds(cx, cy, eye_label, img_np.shape)
            cropped_rgb = img_np[y1_c:y2_c, x1_c:x2_c]

            # 5. Semantic Segmentation Model
            mask_overlay = process_segmentation(cropped_rgb, seg_model)

            # 6. Save Blended Result
            cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
            mask_save_path = os.path.join(CROP_OUTPUT_DIR, f"{base_name}_{eye_label}_mask.png")
            save_overlay_result(cropped_bgr, mask_overlay, mask_save_path)

        except Exception as e:
            print(f"Failed to process {base_name}: {e}")
            traceback.print_exc()

    print("\nAll predictions + directional crops completed successfully.")
    return 0


if __name__ == "__main__":
    main()

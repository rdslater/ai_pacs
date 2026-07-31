"""
test.py

Standalone batch-test harness for the ai_pacs inference pipeline.

This is what the original notebook-derived script did: read a list of
DICOM file paths from an Excel sheet, run the full pipeline on each,
and save the results to disk for manual review. The actual pipeline
logic now lives in inference.py (shared with listener.py) -- this
script is just a thin loop over inference.run_inference() plus
disk I/O, kept around for offline/manual testing against a batch of
files without needing Orthanc running at all.

Usage:
    python test.py

Expects an Excel file (EXCEL_PATH) with a "File Path" column of DICOM
file paths to process.
"""

import os
import traceback

import cv2
import pandas as pd
import pydicom

from inference import InferenceUnavailableError, run_inference

OUTPUT_DIR = "./screenshotscropped"
EXCEL_PATH = r"./data.xlsx"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            ds = pydicom.dcmread(dicom_path)
            result = run_inference(ds)
        except InferenceUnavailableError as e:
            print(f"→ No result: {e}")
            continue
        except Exception as e:
            print(f"Failed to process {base_name}: {e}")
            traceback.print_exc()
            continue

        print(f"→ Classification: {result.eye_label} Eye (yolo_conf={result.yolo_confidence})")
        print(f"→ Crop box: {result.crop_box}")

        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{result.eye_label}_mask.png")
        cv2.imwrite(save_path, result.screenshot)
        print(f"→ Saved overlay to {save_path}")

    print("\nAll predictions + directional crops completed successfully.")
    return 0


if __name__ == "__main__":
    main()

"""
listener.py

DICOM Storage SCP for the ai_pacs project.

Pipeline:

    Orthanc (Docker) --C-STORE--> listener.py --inference.run_inference()--> DICOM-SEG --C-STORE--> Orthanc

Orthanc is configured to forward newly received instances to this
listener as a DICOM "modality". listener.py receives each 2D image
instance, runs it through the real inference pipeline in
inference.py (eye-side classification -> optic disc detection ->
crop -> segmentation), packages the resulting mask as a proper DICOM
Segmentation object (SOP Class "Segmentation Storage") referencing
the source image, and stores that SEG object back into Orthanc. A QA
screenshot (the crop with the mask overlaid) is also saved locally
for review.

Model checkpoints referenced by inference.py are assumed to already
be present in the deployment environment -- see inference.py's
module docstring and README.md for what's needed.

Run:
    python listener.py

Requires an Orthanc instance reachable at ORTHANC_HOST:ORTHANC_PORT,
with this listener registered as a modality so Orthanc knows where to
forward/route studies. See README.md for setup steps.
"""

import logging
import os

import cv2
import numpy as np
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import Verification
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import highdicom as hd

from inference import InferenceUnavailableError, run_inference

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# This listener's own AE identity -- register these with Orthanc as a
# modality so it knows how to reach this script.
LISTENER_AE_TITLE = "AI_PACS"
LISTENER_PORT = 11112

# Orthanc's DICOM (not HTTP) endpoint -- where the resulting SEG
# object gets sent. Defaults assume Orthanc's DICOM port is published
# to the host by Docker (e.g. `-p 4242:4242`).
ORTHANC_AE_TITLE = "ORTHANC"
ORTHANC_HOST = "127.0.0.1"
ORTHANC_PORT = 4242

# Identifies this software as the "device" that created the SEG object.
MANUFACTURER = "ai_pacs"
MODEL_NAME = "ai_pacs-eye-segmentation"
SOFTWARE_VERSION = "0.2.0"
DEVICE_SERIAL_NUMBER = "ai_pacs-listener"

# Where QA overlay screenshots get saved (one per processed instance).
SCREENSHOT_DIR = "./screenshots"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai_pacs.listener")


# --------------------------------------------------------------------------
# DICOM-SEG packaging
# --------------------------------------------------------------------------

def build_segmentation(
    source_ds: Dataset,
    mask: np.ndarray,
    eye_label: str,
    yolo_confidence: float,
) -> hd.seg.Segmentation:
    """Wrap a binary mask + its source image into a DICOM Segmentation object."""
    segment_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label=f"ai_pacs ROI ({eye_label} eye)",
        segmented_property_category=Code("91723000", "SCT", "Anatomical structure"),
        segmented_property_type=Code("91723000", "SCT", "Anatomical structure"),
        algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification=hd.AlgorithmIdentificationSequence(
            name=MODEL_NAME,
            version=SOFTWARE_VERSION,
            family=Code("123109", "DCM", "Manual Processing"),
        ),
    )

    return hd.seg.Segmentation(
        source_images=[source_ds],
        pixel_array=mask,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=[segment_description],
        series_instance_uid=generate_uid(),
        series_number=100,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        manufacturer_model_name=MODEL_NAME,
        software_versions=SOFTWARE_VERSION,
        device_serial_number=DEVICE_SERIAL_NUMBER,
        content_description=f"ai_pacs segmentation ({eye_label} eye, yolo_conf={yolo_confidence})",
    )


def save_screenshot(sop_instance_uid: str, screenshot: np.ndarray) -> str:
    """Saves the QA overlay screenshot to disk, returns the path."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOT_DIR, f"{sop_instance_uid}.jpg")
    cv2.imwrite(path, screenshot)
    return path


# --------------------------------------------------------------------------
# DICOM networking
# --------------------------------------------------------------------------

def send_to_orthanc(ds: Dataset) -> bool:
    """Send a dataset back to Orthanc via C-STORE. Returns True on success."""
    ae = AE(ae_title=LISTENER_AE_TITLE)
    ae.add_requested_context(ds.SOPClassUID, ds.file_meta.TransferSyntaxUID)

    assoc = ae.associate(ORTHANC_HOST, ORTHANC_PORT, ae_title=ORTHANC_AE_TITLE)
    if not assoc.is_established:
        logger.error(
            "Could not associate with Orthanc at %s:%s (AE title %s)",
            ORTHANC_HOST, ORTHANC_PORT, ORTHANC_AE_TITLE,
        )
        return False

    try:
        status = assoc.send_c_store(ds)
        if status and getattr(status, "Status", None) == 0x0000:
            logger.info("Sent SEG instance %s to Orthanc", ds.SOPInstanceUID)
            return True
        logger.error("C-STORE to Orthanc failed, status: %s", status)
        return False
    finally:
        assoc.release()


def ensure_pixel_measures(ds: Dataset) -> Dataset:
    """
    Ensures highdicom can find PixelMeasuresSequence on Ophthalmology/2D datasets.
    """
    # Grab spacing from standard top-level DICOM tags, fallback to 1.0, 1.0 if absent
    pixel_spacing = getattr(ds, "PixelSpacing", getattr(ds, "ImagerPixelSpacing", [1.0, 1.0]))
    slice_thickness = getattr(ds, "SliceThickness", 1.0)

    # Build the required Pixel Measures Dataset
    pixel_measures_item = Dataset()
    pixel_measures_item.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    pixel_measures_item.SliceThickness = float(slice_thickness)

    # 1. Attach directly to top-level if needed
    if "PixelMeasuresSequence" not in ds:
        ds.PixelMeasuresSequence = Sequence([pixel_measures_item])

    # 2. Attach to Shared Functional Groups Sequence (where highdicom checks)
    if "SharedFunctionalGroupsSequence" not in ds:
        shared_item = Dataset()
        shared_item.PixelMeasuresSequence = Sequence([pixel_measures_item])
        ds.SharedFunctionalGroupsSequence = Sequence([shared_item])
    elif "PixelMeasuresSequence" not in ds.SharedFunctionalGroupsSequence[0]:
        ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence = Sequence([pixel_measures_item])

    return ds

def handle_store(event):
    """Handler for evt.EVT_C_STORE -- runs once per received instance."""
    ds = event.dataset
    ds.file_meta = event.file_meta

    logger.info(
        "Received instance %s (Series: %s, Patient ID: %s)",
        ds.SOPInstanceUID,
        getattr(ds, "SeriesDescription", "n/a"),
        getattr(ds, "PatientID", "n/a"),
    )

    if "PixelData" not in ds:
        logger.warning("Instance %s has no PixelData, skipping (not an image)", ds.SOPInstanceUID)
        return 0x0000

    #if ds.pixel_array.ndim != 2:
    #    logger.warning(
    #        "Instance %s is not a single 2D frame (shape %s), skipping -- "
    #        "multi-frame/3D support is not implemented yet",
    #        ds.SOPInstanceUID, ds.pixel_array.shape,
    #    )
    #    return 0x0000

    try:
        result = run_inference(ds)
    except InferenceUnavailableError as e:
        # Expected "nothing found" outcome for some images -- not a failure.
        logger.info("No result for instance %s: %s", ds.SOPInstanceUID, e)
        return 0x0000
    except Exception:
        logger.exception("Inference failed for instance %s", ds.SOPInstanceUID)
        # 0xC001: Unable to process -- tells the sender something went wrong,
        # but we still ack receipt rather than dropping the association.
        return 0xC001

    screenshot_path = save_screenshot(ds.SOPInstanceUID, result.screenshot)
    logger.info(
        "Instance %s: eye=%s, yolo_conf=%s, crop_box=%s, screenshot=%s",
        ds.SOPInstanceUID, result.eye_label, result.yolo_confidence, result.crop_box, screenshot_path,
    )
    # insert spacing if it isn't here
    ds = ensure_pixel_measures(ds)
    try:
        seg = build_segmentation(ds, result.mask, result.eye_label, result.yolo_confidence)
        send_to_orthanc(seg)
        seg.save_as(os.path.join(SCREENSHOT_DIR, f"{ds.SOPInstanceUID}.dcm"))
    except Exception:
        logger.exception("Failed to package/forward SEG for instance %s", ds.SOPInstanceUID)
        return 0xC001

    return 0x0000  # Success


def main():
    ae = AE(ae_title=LISTENER_AE_TITLE)
    ae.supported_contexts = AllStoragePresentationContexts
    ae.add_supported_context(Verification)  # enables C-ECHO for connectivity tests

    handlers = [(evt.EVT_C_STORE, handle_store)]

    logger.info("Starting ai_pacs listener as AE '%s' on port %s ...", LISTENER_AE_TITLE, LISTENER_PORT)
    logger.info(
        "Resulting DICOM-SEG objects will be sent to Orthanc AE '%s' at %s:%s",
        ORTHANC_AE_TITLE, ORTHANC_HOST, ORTHANC_PORT,
    )

    ae.start_server(("0.0.0.0", LISTENER_PORT), evt_handlers=handlers)


if __name__ == "__main__":
    main()

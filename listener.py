"""
listener.py

Basic DICOM Storage SCP for the ai_pacs project.

Pipeline (v2):

    Orthanc (Docker) --C-STORE--> listener.py --segment--> DICOM-SEG --C-STORE--> Orthanc

Orthanc is configured to forward newly received instances to this
listener as a DICOM "modality". listener.py receives each image
instance, runs a placeholder "segmentation" on its pixel data, packages
the result as a proper DICOM Segmentation object (SOP Class
"Segmentation Storage") referencing the source image, and stores that
SEG object back into Orthanc.

`run_inference()` is where a real trained model gets plugged in later.
Right now it's a stand-in: a simple intensity threshold, just to prove
the receive -> infer -> package-as-SEG -> send pipeline works
end-to-end before any real model exists.

Run:
    python listener.py

Requires an Orthanc instance reachable at ORTHANC_HOST:ORTHANC_PORT,
with this listener registered as a modality so Orthanc knows where to
forward/route studies. See README.md for setup steps.
"""

import logging

import numpy as np
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid
from pynetdicom import AE, evt, AllStoragePresentationContexts
from pynetdicom.sop_class import Verification

import highdicom as hd

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
MODEL_NAME = "placeholder-threshold-seg"
SOFTWARE_VERSION = "0.1.0"
DEVICE_SERIAL_NUMBER = "ai_pacs-listener"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai_pacs.listener")


# --------------------------------------------------------------------------
# Inference (placeholder)
# --------------------------------------------------------------------------

def run_inference(pixel_array: np.ndarray) -> np.ndarray:
    """Produce a binary segmentation mask for a single 2D image.

    STAND-IN IMPLEMENTATION. This just thresholds the brightest 25% of
    pixels -- it is not a real segmentation and has no clinical
    meaning. It exists purely to exercise the pipeline (receive ->
    infer -> package as DICOM-SEG -> send) before a trained model is
    wired in here.

    Replace this function's body with real model inference. It should
    keep taking a 2D numpy array and returning a same-shape binary
    (0/1) numpy array.
    """
    threshold = np.percentile(pixel_array, 75)
    return (pixel_array > threshold).astype(np.uint8)


# --------------------------------------------------------------------------
# DICOM-SEG packaging
# --------------------------------------------------------------------------

def build_segmentation(source_ds: Dataset, mask: np.ndarray) -> hd.seg.Segmentation:
    """Wrap a binary mask + its source image into a DICOM Segmentation object."""
    segment_description = hd.seg.SegmentDescription(
        segment_number=1,
        segment_label="ai_pacs placeholder ROI",
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
        content_description="ai_pacs placeholder segmentation",
    )


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

    pixel_array = ds.pixel_array
    if pixel_array.ndim != 2:
        logger.warning(
            "Instance %s is not a single 2D frame (shape %s), skipping -- "
            "multi-frame/3D support is not implemented yet",
            ds.SOPInstanceUID, pixel_array.shape,
        )
        return 0x0000

    try:
        mask = run_inference(pixel_array)
        seg = build_segmentation(ds, mask)
        send_to_orthanc(seg)
    except Exception:
        logger.exception("Failed to process/forward instance %s", ds.SOPInstanceUID)
        # 0xC001: Unable to process -- tells the sender something went wrong,
        # but we still ack receipt rather than dropping the association.
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

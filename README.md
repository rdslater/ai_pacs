# ai_pacs

An AI-in-the-loop PACS pipeline for fundus (retinal) images: receive a
DICOM image from Orthanc, classify eye side, detect the optic disc,
segment a region around it, package the result as a proper DICOM-SEG
object, and send it back.

Claude Code was used to assist as a trial (and MAN it speeds stuff up!)

## Architecture

```
Orthanc (Docker)  --C-STORE-->  listener.py  --inference.run_inference()-->  DICOM-SEG  --C-STORE-->  Orthanc
```

Orthanc is configured to forward newly received image instances to
`listener.py` as if it were another DICOM modality. `listener.py` runs
a Storage SCP (server) that accepts each incoming 2D image, hands it to
`inference.run_inference()`, packages the resulting mask as a proper
DICOM Segmentation object (`highdicom.seg.Segmentation`, SOP Class
"Segmentation Storage") that references the source image, and stores
that SEG object back into Orthanc as a new series. A QA screenshot
(the crop with the predicted mask overlaid in red) is also saved to
`./screenshots/<SOPInstanceUID>.jpg` for review.

Only single-frame 2D images are handled right now (multi-frame/3D
support, and non-image instances like SRs, are logged and skipped).
The original instance sent by Orthanc is left untouched; only the new
SEG object is created and sent back.

### Inference pipeline (`inference.py`)

Given one in-memory DICOM dataset:

1. **Eye side classification** -- a Swin binary classifier labels the
   image Left/Right.
2. **Optic disc detection** -- a YOLO model locates the target region.
   Confidence backs off automatically (0.5 down to 0.001) until a
   detection is found; if nothing is ever found,
   `run_inference()` raises `InferenceUnavailableError` and
   `listener.py` skips that instance (logged, not treated as an error).
3. **Directional crop** -- a 512x512 window is cropped around the
   detection, offset left/right based on eye side.
4. **Segmentation** -- a Unet (mit_b4 encoder) segmentation model
   produces a binary mask over the crop.
5. **Repositioning** -- the crop-space mask is pasted into a
   full-resolution canvas matching the source image, so it lines up
   pixel-for-pixel with the DICOM instance `listener.py` received.
   This is what makes it valid to package as a DICOM-SEG.

`inference.py` loads all three models once (lazily, on first call) and
reuses them for the life of the process rather than reloading per
image.

This is a refactor of an original notebook-style script
(`test.py`) that batch-processed a list of file paths from an Excel
sheet. `test.py` still exists, but now as a thin CLI harness that
reuses `inference.run_inference()` for offline/manual testing against
a batch of files -- it no longer duplicates the pipeline logic.

### Model setup (not included in this repo)

`inference.py` expects a `models.py` in the same directory exposing
`StrongModel` and `SwinBinaryClassifier`, plus three checkpoint files:

| Constant                     | Default filename                                                              | Env var override         |
|-------------------------------|--------------------------------------------------------------------------------|---------------------------|
| `SEGMENTATION_MODEL_PATH`     | `50epochs_strongmodel_augmentation_gencropping_noclahe_8020split_mitUNET.pth` | `AI_PACS_SEG_MODEL_PATH`  |
| `SWIN_MODEL_PATH`              | `swintiny_binary_best.pth`                                                     | `AI_PACS_SWIN_MODEL_PATH` |
| `YOLO_MODEL_PATH`              | `Yolo1.pt`                                                                      | `AI_PACS_YOLO_MODEL_PATH` |

None of these (models.py, checkpoints, or `data.xlsx` for `test.py`)
are committed to this repo -- add them locally before running.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Note: `torch`/`torchvision`/`ultralytics` are sizeable installs and,
if you have a GPU, you'll likely want CUDA-matched wheels rather than
the generic ones `pip install -r requirements.txt` will pull in --
check https://pytorch.org for the right index URL for your setup.

### 2. Add model files

Place `models.py` (defining `StrongModel` and `SwinBinaryClassifier`)
and the three checkpoint files alongside `listener.py`, or set the
`AI_PACS_SEG_MODEL_PATH` / `AI_PACS_SWIN_MODEL_PATH` /
`AI_PACS_YOLO_MODEL_PATH` environment variables to point elsewhere.

### 3. Confirm Orthanc's DICOM port is reachable

Your Orthanc container needs its DICOM port (default `4242`) published
to the host, e.g.:

```bash
docker run -p 4242:4242 -p 8042:8042 jodogne/orthanc
```

Edit `ORTHANC_HOST` / `ORTHANC_PORT` / `ORTHANC_AE_TITLE` at the top of
`listener.py` if your setup differs from the defaults (`127.0.0.1:4242`,
AE title `ORTHANC`).

### 4. Register listener.py as a modality in Orthanc

Orthanc needs to know this listener exists so it can route instances
to it. Easiest way is via Orthanc's REST API (works whether or not
Orthanc's config file is easily editable inside the container):

```bash
curl -X PUT http://localhost:8042/modalities/ai_pacs \
  -d '{
    "AET": "AI_PACS",
    "Host": "host.docker.internal",
    "Port": 11112,
    "Manufacturer": "Generic"
  }'
```

Notes:
- `AET` and `Port` must match `LISTENER_AE_TITLE` / `LISTENER_PORT` in `listener.py`.
- `Host` is Orthanc's view of where to reach the listener. If
  `listener.py` runs on the Docker host (not inside a container),
  `host.docker.internal` usually works on Docker Desktop (Mac/Windows).
  On Linux you may need the host's actual IP, or run Orthanc with
  `--network host`.
- Replace `localhost:8042` with wherever Orthanc's HTTP API is
  actually exposed.

This registers the modality but does **not** auto-forward anything yet
-- that's manual for now (step 6), which is fine for this milestone.

### 5. Run the listener

```bash
python listener.py
```

You should see it log that it started and which Orthanc it will send
results back to. Model loading happens lazily on the first received
instance, not at startup, so the first C-STORE will be slower than
subsequent ones.

### 6. Send it a test instance

Manually, from Orthanc Explorer: open any study/instance -> "Send to
modality" -> pick `ai_pacs`.

Or from the command line, without needing Orthanc at all yet, using
pynetdicom's own test client against the listener directly (use a
single-frame fundus image -- multi-frame objects aren't handled yet):

```bash
python -m pynetdicom storescu 127.0.0.1 11112 /path/to/some.dcm -aet TESTSCU -aec AI_PACS
```

Watch `listener.py`'s console output -- you should see it log the
received instance, run inference, build the SEG object, save a QA
screenshot, then attempt (and, once Orthanc is reachable, succeed) to
send the SEG back.

## Roadmap

This is deliberately the smallest possible working slice. Next steps,
roughly in order:

1. ~~Automate the Orthanc -> listener forwarding (Orthanc Lua script
   `OnStableStudy`/`OnStoredInstance` calling `SendToModality`, or the
   Orthanc auto-routing plugin) so instances flow through without a
   manual "Send to modality" click.~~ 
2. ~~Validate the real model checkpoints end-to-end (this was built and
   smoke-tested with stand-in/mocked models -- no checkpoints were
   available in the environment this was written in).~~
3. Handle multi-frame/3D source images, not just single 2D frames. IN PROGRESS-TESTING
4. Add error handling / retry logic for the outbound C-STORE.
5. Containerize `listener.py` (with model weights) so it runs
   alongside Orthanc via `docker-compose`.
6. Add tests (a fake Orthanc SCP to assert what gets sent back; assert
   the SEG references the right source image/series).
7. Add config via environment variables / a config file instead of
   constants at the top of `listener.py`.

## Issues
- Requirements.txt is giving a little trouble especially working with conda.  May need to manually adjust to torch-cpu
- Found a problem where DICOM-SEG is expected to be 2D, but Ophthamology often puts 2D images in as color (3D!) Testing a fix currently.

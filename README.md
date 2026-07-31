# ai_pacs

An AI-in-the-loop PACS pipeline. Built top-down: this milestone proves
the full plumbing works -- receive a DICOM image from Orthanc, run
inference on it, package the result as a proper DICOM-SEG object, and
send it back -- with a placeholder threshold standing in for a real
trained model.

## Architecture (v2)

```
Orthanc (Docker)  --C-STORE-->  listener.py  --infer + package as DICOM-SEG-->  --C-STORE-->  Orthanc
```

Orthanc is configured to forward newly received image instances to
`listener.py` as if it were another DICOM modality. `listener.py`
runs a Storage SCP (server) that accepts each incoming 2D image,
runs `run_inference()` on its pixel data, packages the result as a
proper DICOM Segmentation object (`highdicom.seg.Segmentation`, SOP
Class "Segmentation Storage") that references the source image, and
stores that SEG object back into Orthanc as a new series.

`run_inference()` in `listener.py` is where a real trained model gets
plugged in later. Right now it's a stand-in: a simple intensity
threshold (brightest 25% of pixels) with no clinical meaning. It
exists purely to exercise receive -> infer -> package-as-SEG -> send
before a real model is wired in. Swap its body out for real inference
whenever the model is ready -- it should keep taking a 2D numpy array
in and returning a same-shape binary (0/1) numpy array out.

Only single-frame 2D images are handled right now (multi-frame/3D
support, and non-image instances like SRs, are logged and skipped).
The original instance sent by Orthanc is left untouched; only the new
SEG object is created and sent back.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Confirm Orthanc's DICOM port is reachable

Your Orthanc container needs its DICOM port (default `4242`) published
to the host, e.g.:

```bash
docker run -p 4242:4242 -p 8042:8042 jodogne/orthanc
```

Edit `ORTHANC_HOST` / `ORTHANC_PORT` / `ORTHANC_AE_TITLE` at the top of
`listener.py` if your setup differs from the defaults (`127.0.0.1:4242`,
AE title `ORTHANC`).

### 3. Register listener.py as a modality in Orthanc

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
-- that's manual for now (step 4), which is fine for this milestone.

### 4. Run the listener

```bash
python listener.py
```

You should see it log that it started and which Orthanc it will send
results back to.

### 5. Send it a test instance

Manually, from Orthanc Explorer: open any study/instance -> "Send to
modality" -> pick `ai_pacs`.

Or from the command line, without needing Orthanc at all yet, using
pynetdicom's own test client against the listener directly (use a
single-frame image such as a CT/MR/CR slice -- multi-frame objects
aren't handled yet):

```bash
python -m pynetdicom storescu 127.0.0.1 11112 /path/to/some.dcm -aet TESTSCU -aec AI_PACS
```

Watch `listener.py`'s console output -- you should see it log the
received instance, build the SEG object, then attempt (and, once
Orthanc is reachable, succeed) to send it back.

## Roadmap

This is deliberately the smallest possible working slice. Next steps,
roughly in order:

1. Automate the Orthanc -> listener forwarding (Orthanc Lua script
   `OnStableStudy`/`OnStoredInstance` calling `SendToModality`, or the
   Orthanc auto-routing plugin) so instances flow through without a
   manual "Send to modality" click.
2. Replace the placeholder threshold in `run_inference()` with a real
   trained model.
3. Handle multi-frame/3D source images, not just single 2D frames.
4. Add error handling / retry logic for the outbound C-STORE.
5. Containerize `listener.py` so it runs alongside Orthanc via
   `docker-compose`.
6. Add tests (a fake Orthanc SCP to assert what gets sent back; assert
   the SEG references the right source image/series).
7. Add config via environment variables / a config file instead of
   constants at the top of `listener.py`.

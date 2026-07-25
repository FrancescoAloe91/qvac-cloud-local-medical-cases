# QVAC SDK sidecar

Local HTTP wrapper around [`@qvac/sdk`](https://docs.qvac.tether.io/quickstart/) so the Python benchmark can call MedPsy on-device via the QVAC SDK.

## Hardware

The SDK **defaults to GPU** (`device: gpu`, `gpu_layers: 99`) — on Apple Silicon that means **Metal**, not forced CPU.

The old Ollama path used `num_gpu 0` only as a workaround for a Metal crash on some Macs. This sidecar prefers GPU and **falls back to CPU only if GPU load fails**.

```bash
# defaults (GPU / Metal)
cd sidecar && npm start

# force CPU if needed
QVAC_DEVICE=cpu QVAC_GPU_LAYERS=0 npm start

# prefer discrete GPU when present
QVAC_MAIN_GPU=dedicated npm start
```

Check what actually loaded:

```bash
curl -s http://127.0.0.1:8787/health
# → device, gpu_layers, modelLoaded
```

## Setup

```bash
# Node.js >= 22.17
../scripts/setup_qvac_sidecar.sh
# or: npm install
```

Place the MedPsy GGUF at `../models/medpsy-4b-q4_k_m-imat.gguf` (or set `QVAC_MODEL_PATH`).

## Run

```bash
npm start
# GET  http://127.0.0.1:8787/health
# POST http://127.0.0.1:8787/generate  {"prompt":"..."}
```

Model **warm-loads at startup** (set `QVAC_WARM_LOAD=0` to defer).  
`QVAC_PREDICT` caps max new tokens (default 3000, aligned with cloud candidates).

### macOS dependency

The SDK worker (`.bare` addons) links against **OpenSSL 3**.

`./scripts/setup_qvac_sidecar.sh` will:

1. `npm install`
2. Build/link OpenSSL 3 under `~/.local/qvac-openssl` (or use Homebrew if present)
3. Rewrite QVAC `.bare` install names + create a space-free model symlink

```bash
../scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

Without OpenSSL 3 the worker aborts on load (`libssl.3.dylib` missing).  
MedPsy in this demo uses the **QVAC SDK only** (no Ollama path).

The benchmark auto-detects this sidecar; if it is down, QVAC is skipped.

## Live streaming

`POST /generate/stream` returns NDJSON lines:

```json
{"type":"token","token":"..."}
{"type":"done","content":"...","ttft_s":0.4,"tps":32,"device":"gpu",...}
```

Used by **Run QVAC only · $0** in the demo UI so MedPsy text appears token-by-token.

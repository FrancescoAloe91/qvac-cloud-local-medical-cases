# MedPsy GGUF (not stored in git)

Weights are **not in git** (GitHub file-size limit). Download locally.

| File | Role | ~Size | Hugging Face |
|------|------|-------|----------------|
| `medpsy-1.7b-q4_k_m-imat.gguf` | Smaller / phone-class | ~1.3 GB | [`MedPsy-1.7B-GGUF`](https://huggingface.co/qvac/MedPsy-1.7B-GGUF) |
| `medpsy-4b-q4_k_m-imat.gguf` | Default sidecar (balanced) | ~2.5 GB | [`MedPsy-4B-GGUF`](https://huggingface.co/qvac/MedPsy-4B-GGUF) |
| `medpsy-4b-q8_0.gguf` | Higher-quality 4B quant | ~4.4 GB | same 4B-GGUF repo |

```bash
# default 4B Q4
./scripts/download_medpsy_gguf.sh

# optional variants
MEDPSY_QUANT=medpsy-4b-q8_0.gguf ./scripts/download_medpsy_gguf.sh
MEDPSY_REPO=qvac/MedPsy-1.7B-GGUF MEDPSY_QUANT=medpsy-1.7b-q4_k_m-imat.gguf ./scripts/download_medpsy_gguf.sh

# or full local install:
./install.sh
```

In the Streamlit app, **3× QVAC compare** hot-swaps these three GGUFs via sidecar `POST /load`
(standard mode keeps only 4B Q4). Restart the sidecar after pulling this feature so `/load` exists.

Point the sidecar at a file with `QVAC_MODEL_PATH=/path/to/….gguf`, then:

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

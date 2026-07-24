# MedPsy GGUF (not stored in git)

The on-device model file is **~2.5 GB** (`medpsy-4b-q4_k_m-imat.gguf`).  
GitHub rejects files that large, so it is **downloaded on install**, not committed.

```bash
# from repo root
./scripts/download_medpsy_gguf.sh
# or full local install:
./install.sh
```

Source: Hugging Face [`qvac/MedPsy-4B-GGUF`](https://huggingface.co/qvac/MedPsy-4B-GGUF).

After download, start the QVAC SDK sidecar:

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

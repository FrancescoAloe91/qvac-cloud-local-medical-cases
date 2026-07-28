# On-device GGUFs (not stored in git)

Weights are **downloaded from Hugging Face** into `models/`. They are **never**
committed to GitHub (size + license hygiene). Your private run History under
`artifacts/` is also **never** in git — cloning the repo does not give anyone
your prompts, outputs, scores, or means.

SHA pin is **optional**: leave `MEDPSY_GGUF_SHA256` unset for a convenience
download, or set it (and peer digests via the download scripts) to bit-pin the
quant used for Band B / MedPsy reproducibility.

## One-shot: full pack (~14 GB)

All three MedPsy quants used by **3× QVAC** + Band B peers (Gemma / Llama / Phi):

```bash
./scripts/download_all_ggufs.sh
# or: FULL_MODELS=1 ./install.sh
```

## MedPsy (QVAC brand)

| File | Role | ~Size | Hugging Face |
|------|------|-------|----------------|
| `medpsy-1.7b-q4_k_m-imat.gguf` | Smaller / phone-class | ~1.3 GB | [`MedPsy-1.7B-GGUF`](https://huggingface.co/qvac/MedPsy-1.7B-GGUF) |
| `medpsy-4b-q4_k_m-imat.gguf` | Default sidecar (balanced) | ~2.5 GB | [`MedPsy-4B-GGUF`](https://huggingface.co/qvac/MedPsy-4B-GGUF) |
| `medpsy-4b-q8_0.gguf` | Higher-quality 4B quant | ~4.4 GB | same 4B-GGUF repo |

```bash
./scripts/download_medpsy_gguf.sh
MEDPSY_QUANT=medpsy-4b-q8_0.gguf ./scripts/download_medpsy_gguf.sh
MEDPSY_REPO=qvac/MedPsy-1.7B-GGUF MEDPSY_QUANT=medpsy-1.7b-q4_k_m-imat.gguf ./scripts/download_medpsy_gguf.sh
```

## Band B · open local peers (Q4 GGUF)

Same privacy as MedPsy (local GGUF). Peer class for fair on-device vs QVAC compare.

| File | Role | ~Size | Hugging Face |
|------|------|-------|----------------|
| `gemma-2-2b-it-Q4_K_M.gguf` | Google open (Gemma-2-2B-IT) | ~1.6 GB | [`gemma-2-2b-it-GGUF`](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Meta open | ~2.0 GB | [`Llama-3.2-3B-Instruct-GGUF`](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| `Phi-3.5-mini-instruct-Q4_K_M.gguf` | Microsoft open | ~2.2 GB | [`Phi-3.5-mini-instruct-GGUF`](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

```bash
./scripts/download_local_peers.sh
```

In the Streamlit app, Band B + MedPsy share one sidecar (serial GGUF load). Toggle **3× QVAC** adds the three MedPsy quants (grid 3×3).

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

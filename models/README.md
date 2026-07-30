# On-device GGUFs (not stored in git)

Weights are **downloaded from Hugging Face** into `models/`. They are **never**
committed to GitHub (size + license hygiene). Your private run History under
`artifacts/` is also **never** in git — cloning the repo does not give anyone
your prompts, outputs, scores, or means.

SHA pin is **optional**: leave `MEDPSY_GGUF_SHA256` unset for a convenience
download, or set it (and peer digests via the download scripts) to bit-pin the
quant used for Band B / MedPsy / medical_local reproducibility.

## One-shot: full pack (~25 GB)

All three MedPsy quants used by **3× QVAC** + Band B generics + medical_local
peers (MedGemma / BioMistral / OpenBioLLM):

```bash
./scripts/download_all_ggufs.sh
# or: FULL_MODELS=1 ./install.sh
# MedPsy + Band B only: SKIP_MEDICAL=1 ./scripts/download_all_ggufs.sh
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

Same privacy as MedPsy (local GGUF). Generic peer class for fair on-device vs QVAC
compare — **not** medical-specialized.

| File | Role | ~Size | Hugging Face |
|------|------|-------|----------------|
| `gemma-2-2b-it-Q4_K_M.gguf` | Google open (Gemma-2-2B-IT) | ~1.6 GB | [`gemma-2-2b-it-GGUF`](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF) |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Meta open | ~2.0 GB | [`Llama-3.2-3B-Instruct-GGUF`](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| `Phi-3.5-mini-instruct-Q4_K_M.gguf` | Microsoft open | ~2.2 GB | [`Phi-3.5-mini-instruct-GGUF`](https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF) |

```bash
./scripts/download_local_peers.sh
```

## Band medical_local · medical-specialized peers (Q4_K_M)

Distinct from Band B generics. Toggle **Medical local peers** in the UI, or use
preset **Solo locali medici** (3 MedPsy + 3 medical, cloud/generics off).

| File (local name) | Role | HF repo · remote file |
|-------------------|------|------------------------|
| `medgemma-4b-it-Q4_K_M.gguf` | MedGemma 4B IT | [`unsloth/medgemma-4b-it-GGUF`](https://huggingface.co/unsloth/medgemma-4b-it-GGUF) · `medgemma-4b-it-Q4_K_M.gguf` |
| `BioMistral-7B-Q4_K_M.gguf` | BioMistral 7B | [`BioMistral/BioMistral-7B-GGUF`](https://huggingface.co/BioMistral/BioMistral-7B-GGUF) · `ggml-model-Q4_K_M.gguf` |
| `Llama3-OpenBioLLM-8B.Q4_K_M.gguf` | OpenBioLLM 8B | [`QuantFactory/Llama3-OpenBioLLM-8B-GGUF`](https://huggingface.co/QuantFactory/Llama3-OpenBioLLM-8B-GGUF) · `Llama3-OpenBioLLM-8B.Q4_K_M.gguf` |

```bash
./scripts/download_medical_peers.sh
```

In the Streamlit app, all on-device bands share one sidecar (serial GGUF load).
Toggle **3× QVAC** adds the three MedPsy quants. Full grid with medical peers
on = up to **12** models (3 cloud + 3 generic + 3 medical + 3 MedPsy).

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

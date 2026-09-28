# PD-WM: LLM-Guided Multimodal World Model for Parkinson’s Severity and Personalized Management Profiling

This repository provides a clean structural reference implementation of **PD-WM: LLM-Guided Multimodal World Model for Parkinson's Severity and Personalized Management Profiling**, plus a lightweight end-to-end verification mode. It covers patient matching, leakage-safe labels and narratives, multimodal bags, PSTAL, mixture-of-experts fusion, ordinal severity prediction, an action-conditioned structured energy transition, management slot prediction, controlled verbalization, evaluation, plots, and saved results.

## Important reproducibility notice

The default `QUICK_RUN` configuration uses lightweight interface-compatible placeholder implementations for the frozen I-JEPA and MedCPT encoders. This verifies the complete software and data flow without downloading or distributing large pretrained models. It is **not intended to reproduce manuscript performance**, and quick-run metrics from a tiny split must not be interpreted as clinical or manuscript-level results.

 

## Quick start

Use Python 3.10+ from this directory:

```bash
pip install -r requirements.txt
python run_quick_test.py
```

The default run uses at most 10 subjects, 3 epochs, deterministic seed 42, small batches, and `num_workers=0` for Windows. Change `MAX_SUBJECTS` to 5, 10, 15, or 20 in `config.py`. Results are written to `outputs/`; caches are written to `cache/`. Both are intentionally ignored by Git.

Expected local layout:

```text
parent/
  OpenNeuro_PD_data/
  neuroplan_pd_patient_master_with_severity.csv
  api_key.txt                         # optional, ignored
  pd_wm_reference/
    run_quick_test.py
```

## Dataset behavior and scientific safeguards

Subjects are joined by normalized numeric CSV `subject_id` and BIDS `sub-XX`. Splits occur by patient before the training candidate set is built. Candidate management states are unique structured profiles from the **training split only**. The severity rule is: non-PD = 0; PD stage I–II = 1; PD stage III+ = 2; UPDRS-III `<=32` versus `>32` is used only when stage is unavailable.

The supplied local OpenNeuro snapshot contains T1-weighted anatomical MRI and resting-state fMRI, but no files labelled as DaT/SPECT. The CSV's historical `dat_image_count` values therefore do not describe files presently under that root. This implementation does not relabel fMRI as DaT: it records the mismatch in `outputs/dataset_audit.json` and sends an explicit zero/masked bag through the DaT branch. When genuine DaT/SPECT files with `dat` or `spect` in their filename are added under subject folders, discovery uses them automatically.

The existing CSV `patient_narrative` is not used because it contains diagnosis, stage, and UPDRS-III. Narratives are rebuilt only from the allow-listed clinical fields in `pdwm/data.py`. Action descriptors receive only that safe narrative and predefined medicine identity. The controlled verbalizer receives only predictions, never ground-truth management targets.

## Placeholder and real encoders

By default, `IJEPAVisionEncoder` is a small trainable CNN and `MedCPTTextEncoder` is a deterministic fixed-dimensional hashing encoder. Saved configuration records both backends as `placeholder`; these features must not be described as real I-JEPA or MedCPT features.

### Real I-JEPA

1. Place the intended checkpoint under `weights/ijepa/` (the configured marker is `weights/ijepa/PUT_IJEPA_WEIGHTS_HERE`).
2. Implement the checkpoint/model loading branch in `IJEPAVisionEncoder` in `pdwm/encoders.py`, preserving its `[images] -> [image embeddings]` interface and `VISUAL_DIM`.
3. Set `IJEPA_BACKEND = "real"` and `IJEPA_CHECKPOINT` to the local checkpoint in `config.py`.

The reference deliberately raises `NotImplementedError` for `real` until the appropriate upstream I-JEPA implementation and checkpoint contract are supplied; it never silently claims placeholder features are I-JEPA.

### Real MedCPT

1. Place a local MedCPT model and tokenizer under `weights/medcpt/` (the configured marker is `weights/medcpt/PUT_MEDCPT_MODEL_HERE`).
2. Implement the local model-loading branch in `MedCPTTextEncoder` in `pdwm/encoders.py`, preserving `encode(list[str]) -> Tensor[n, TEXT_DIM]` (or add a projection to `TEXT_DIM`).
3. Set `MEDCPT_BACKEND = "real"` and `MEDCPT_MODEL_PATH` to that directory in `config.py`.

No pretrained weights are downloaded by the quick run.

## Optional OpenAI narratives

The default is `USE_LLM = False`, so the repository runs offline with deterministic target-safe templates. To enable cached GPT generation, set `USE_LLM = True` and provide `OPENAI_API_KEY` in the environment or put the key only in the parent-level `api_key.txt`. The key is never printed or copied. `api_key.txt`, `.env`, and key files are ignored.

Every successful **or failed** API attempt executes `time.sleep(max(5, LLM_SLEEP_SECONDS))`; responses are cached and never regenerated when cached. If the API is absent or fails, generation falls back deterministically and training continues.

## Outputs

The run creates:

```text
outputs/
  dataset_audit.json
  subject_manifest.csv
  image_manifest.csv
  split_manifest.csv
  training_history.csv
  severity_metrics.json
  management_metrics.json
  predictions.csv
  router_weights.csv
  candidate_management_states.csv
  example_verbalizations.json
  run_config.json
  environment_info.txt
  best_model.pt
  figures/
```


# Cite

Please cite the following paper:

## Title:
PD-WM: LLM-Guided Multimodal World Model for Parkinson’s Severity and Personalized Management Profiling

Muhammad Ayoub, Hai Zhao, and Yi Zhao
Accepted as a Regular paper at the 2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2026).

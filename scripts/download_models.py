#!/usr/bin/env python3
"""
Download the reference models used by the ChronoScribe implementation.

Models:
  - Qwen3.5-9B (~20 GB on disk) — reference language controller
  - Chronos-2 (0.5 GB) — reference time-series forecasting specialist

The fine-tuned LoRA adapter (checkpoint-7500) is NOT downloaded —
it must be placed manually in models/checkpoint-7500/.

Usage:
  python scripts/download_models.py              # download both
  python scripts/download_models.py --qwen-only  # Qwen3.5-9B only
  python scripts/download_models.py --c2-only    # Chronos-2 only
  python scripts/download_models.py --force      # re-download even if exists
"""
import json
import os, sys, argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")

QWEN_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN_DIR = os.path.join(MODELS_DIR, "Qwen3.5-9B")

CHRONOS2_MODEL_ID = "amazon/chronos-2"
CHRONOS2_DIR = os.path.join(MODELS_DIR, "chronos-2")

# Pin the model snapshots used by the documented reference setup. Do not follow
# `main` implicitly: a moving upstream revision breaks exact reproduction.
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CHRONOS2_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def _metadata_path(model_dir):
    return os.path.join(model_dir, "chronoscribe_download.json")


def _matches_reference(model_dir, model_id, revision):
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        return False
    try:
        with open(_metadata_path(model_dir), encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata == {"model_id": model_id, "revision": revision}
    except (OSError, ValueError):
        return False


def _write_metadata(model_dir, model_id, revision):
    with open(_metadata_path(model_dir), "w", encoding="utf-8") as handle:
        json.dump({"model_id": model_id, "revision": revision}, handle, indent=2)
        handle.write("\n")


def download_qwen(force=False):
    """Download the reference language controller without bundling credentials."""
    if _matches_reference(QWEN_DIR, QWEN_MODEL_ID, QWEN_REVISION) and not force:
        size = _dir_size(QWEN_DIR)
        print(f"✅ Qwen3.5-9B already exists ({size:.1f} GB), skipping. Use --force to re-download.")
        return

    print(f"⬇  Downloading {QWEN_MODEL_ID} (~20 GB, revision {QWEN_REVISION[:12]})...")
    os.makedirs(QWEN_DIR, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=QWEN_MODEL_ID,
            local_dir=QWEN_DIR,
            revision=QWEN_REVISION,
            max_workers=4,
        )
    except ImportError:
        print("huggingface_hub not available, trying transformers fallback...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, revision=QWEN_REVISION, torch_dtype="auto")
        model.save_pretrained(QWEN_DIR)
        tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, revision=QWEN_REVISION, trust_remote_code=True)
        tok.save_pretrained(QWEN_DIR)

    _write_metadata(QWEN_DIR, QWEN_MODEL_ID, QWEN_REVISION)
    size = _dir_size(QWEN_DIR)
    print(f"✅ Qwen3.5-9B downloaded ({size:.1f} GB) → {QWEN_DIR}")


def download_chronos2(force=False):
    """Download the reference forecasting specialist from its public source."""
    if _matches_reference(CHRONOS2_DIR, CHRONOS2_MODEL_ID, CHRONOS2_REVISION) and not force:
        size = _dir_size(CHRONOS2_DIR)
        print(f"✅ Chronos-2 already exists ({size:.1f} GB), skipping. Use --force to re-download.")
        return

    print(f"⬇  Downloading {CHRONOS2_MODEL_ID} (~0.5 GB, revision {CHRONOS2_REVISION[:12]})...")
    os.makedirs(CHRONOS2_DIR, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=CHRONOS2_MODEL_ID,
            local_dir=CHRONOS2_DIR,
            revision=CHRONOS2_REVISION,
            max_workers=4,
        )
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download Chronos-2 into the repository-local models directory."
        )

    _write_metadata(CHRONOS2_DIR, CHRONOS2_MODEL_ID, CHRONOS2_REVISION)
    size = _dir_size(CHRONOS2_DIR)
    print(f"✅ Chronos-2 downloaded ({size:.1f} GB) → {CHRONOS2_DIR}")


def _dir_size(path):
    """Return directory size in GB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description="Download ChronoScribe reference models")
    parser.add_argument("--qwen-only", action="store_true")
    parser.add_argument("--c2-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.qwen_only:
        download_qwen(args.force)
    elif args.c2_only:
        download_chronos2(args.force)
    else:
        download_qwen(args.force)
        print()
        download_chronos2(args.force)

    print(f"\n📁 models/ directory ready.")
    print(f"   Next: place your LoRA adapter in models/checkpoint-7500/")
    print("   Then: python runners/run_chronoscribe.py --manifest benchmark/evaluation_manifest.json")


if __name__ == "__main__":
    main()

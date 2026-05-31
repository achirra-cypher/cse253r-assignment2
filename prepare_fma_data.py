#!/usr/bin/env python3
"""
prepare_fma_data.py — Download FMA-small from HuggingFace and prepare AudioCraft
training pairs for MusicGen fine-tuning (Task 4, Day 3).

Produces:
  musicgen_data/
    train/{genre}/*.wav + *.json   — audio + metadata (description field)
    valid/{genre}/*.wav + *.json
  musicgen_data/dataset_info.json  — split counts and genre mapping
  config/dset/audio/fma_genre.yaml — AudioCraft datasource config (for fine-tuning)

Usage:
  pip install datasets soundfile librosa
  python prepare_fma_data.py                    # full: 4 genres × 200 tracks
  python prepare_fma_data.py --n-samples 20     # quick smoke test
  python prepare_fma_data.py --genres Hip-Hop Folk

After running, create AudioCraft manifest files (requires audiocraft installed):
  python -m audiocraft.data.audio_dataset musicgen_data/train egs/fma_genre/train/data.jsonl.gz
  python -m audiocraft.data.audio_dataset musicgen_data/valid egs/fma_genre/valid/data.jsonl.gz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np

# Target genres for fine-tuning (4 of 8 FMA genres — diverse styles)
DEFAULT_GENRES = ["Hip-Hop", "Folk", "Electronic", "Rock"]

GENRE_PROMPTS = {
    "Hip-Hop": "hip hop music with beats and rhythm",
    "Pop": "upbeat pop music with melody",
    "Folk": "acoustic folk music with guitar",
    "Experimental": "experimental avant-garde music",
    "Rock": "energetic rock music with electric guitar",
    "International": "world music with traditional instruments",
    "Electronic": "electronic music with synthesizers",
    "Instrumental": "instrumental music without vocals",
}

TARGET_SAMPLE_RATE = 32000  # MusicGen expects 32 kHz

# HF dataset may store genre as string name OR int label
GENRE_ID_TO_NAME = {
    0: "Electronic",
    1: "Experimental",
    2: "Folk",
    3: "Hip-Hop",
    4: "Instrumental",
    5: "International",
    6: "Pop",
    7: "Rock",
}


def _normalize_genre(value) -> str:
    if isinstance(value, str):
        return value
    return GENRE_ID_TO_NAME.get(int(value), str(value))


def _load_audio_from_row(row: dict) -> tuple[np.ndarray, int]:
    """Load audio from a HF dataset row without torchcodec (Colab-safe)."""
    import io
    import librosa
    import soundfile as sf

    audio = row["audio"]
    if isinstance(audio, dict):
        if "array" in audio and audio["array"] is not None:
            arr = np.array(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate") or TARGET_SAMPLE_RATE)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            return arr, sr
        if audio.get("bytes"):
            buf = io.BytesIO(audio["bytes"])
            arr, sr = librosa.load(buf, sr=None, mono=True)
            return arr.astype(np.float32), int(sr)
        if audio.get("path"):
            arr, sr = librosa.load(audio["path"], sr=None, mono=True)
            return arr.astype(np.float32), int(sr)
    raise ValueError("No decodable audio in row")


def _genre_indices(ds, genre: str) -> list[int]:
    """Return row indices for a genre without decoding audio."""
    genres = [_normalize_genre(g) for g in ds["genre"]]
    return [i for i, g in enumerate(genres) if g == genre]


def _write_metadata_json(path: Path, genre: str, track_id: int, title: str, artist: str):
    """Write AudioCraft-compatible metadata JSON alongside each audio file."""
    meta = {
        "key": "",
        "artist": artist or "Unknown",
        "sample_rate": TARGET_SAMPLE_RATE,
        "file_extension": "wav",
        "description": GENRE_PROMPTS.get(genre, f"{genre} music"),
        "keywords": genre.lower(),
        "duration": 30.0,
        "bpm": "",
        "genre": genre.lower(),
        "title": title or f"track_{track_id}",
        "name": path.stem,
        "instrument": "",
        "moods": [],
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _resample_and_save(audio_array: np.ndarray, orig_sr: int, out_wav: Path):
    """Resample to 32 kHz mono and save as WAV."""
    import librosa
    import soundfile as sf

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=0)

    if orig_sr != TARGET_SAMPLE_RATE:
        audio_array = librosa.resample(
            audio_array.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )

    # Normalize to [-1, 1]
    peak = np.max(np.abs(audio_array)) + 1e-8
    audio_array = audio_array / peak * 0.95

    sf.write(str(out_wav), audio_array, TARGET_SAMPLE_RATE)


def download_and_prepare(
    genres: list[str],
    n_samples: int,
    output_dir: str = "musicgen_data",
    seed: int = 42,
    hf_split: str = "train",
):
    """Download FMA from HuggingFace and write (audio, description) pairs."""
    from datasets import Audio, load_dataset

    print("=" * 60)
    print("Loading FMA-small from HuggingFace (rpmon/fma-genre-classification)")
    print("=" * 60)

    ds = load_dataset("rpmon/fma-genre-classification", split=hf_split)
    # Avoid torchcodec decode on full-dataset iteration (breaks on corrupt tracks in Colab)
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  Loaded {len(ds)} tracks from split='{hf_split}'")

    sample_genres = sorted(set(_normalize_genre(g) for g in ds["genre"][:20]))
    print(f"  Genre labels (sample): {sample_genres}")

    rng = random.Random(seed)
    out_root = Path(output_dir)
    train_root = out_root / "train"
    valid_root = out_root / "valid"

    stats = {"train": {}, "valid": {}, "genres": genres, "n_samples_per_genre": n_samples}

    for genre in genres:
        indices = _genre_indices(ds, genre)
        if len(indices) < n_samples:
            raise ValueError(
                f"Genre '{genre}' has only {len(indices)} tracks, "
                f"need {n_samples}. Try a smaller --n-samples."
            )

        rng.shuffle(indices)
        selected_indices = indices[: n_samples * 2]  # extra buffer for corrupt files

        saved_rows = []
        skipped = 0
        for idx in selected_indices:
            if len(saved_rows) >= n_samples:
                break
            try:
                row = ds[int(idx)]
                row = dict(row)
                row["genre"] = _normalize_genre(row["genre"])
                arr, sr = _load_audio_from_row(row)
                row["_audio_array"] = arr
                row["_audio_sr"] = sr
                saved_rows.append(row)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    skipped track idx={idx} ({e})")
                continue

        if len(saved_rows) < n_samples:
            raise ValueError(
                f"Genre '{genre}': only {len(saved_rows)}/{n_samples} tracks decoded "
                f"({skipped} skipped as corrupt). Try a different --seed."
            )
        if skipped:
            print(f"  {genre}: skipped {skipped} corrupt/unreadable tracks")

        selected = saved_rows[:n_samples]

        # 90/10 train/valid within each genre subset
        n_train = int(0.9 * len(selected))
        train_rows = selected[:n_train]
        valid_rows = selected[n_train:]

        for split_name, rows, split_root in [
            ("train", train_rows, train_root),
            ("valid", valid_rows, valid_root),
        ]:
            genre_dir = split_root / genre.replace(" ", "_")
            genre_dir.mkdir(parents=True, exist_ok=True)

            for i, row in enumerate(rows):
                track_id = row.get("track_id", i)
                title = row.get("title", "")
                artist = row.get("artist", "")

                stem = f"{genre.replace(' ', '_')}_{track_id:05d}"
                out_wav = genre_dir / f"{stem}.wav"
                out_json = genre_dir / f"{stem}.json"

                _resample_and_save(row["_audio_array"], row["_audio_sr"], out_wav)
                _write_metadata_json(out_json, genre, track_id, title, artist)

            stats[split_name][genre] = len(rows)
            print(f"  {split_name}/{genre}: {len(rows)} tracks → {genre_dir}")

    # Save dataset info
    info_path = out_root / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ {info_path}")

    _write_audiocraft_config(out_root)
    return stats


def _write_audiocraft_config(out_root: Path):
    """Write AudioCraft datasource YAML for fine-tuning."""
    config_dir = Path("config/dset/audio")
    config_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = """# @package __global__
# Auto-generated by prepare_fma_data.py — FMA genre fine-tuning datasource

datasource:
  max_sample_rate: 32000
  max_channels: 1

  train: egs/fma_genre/train
  valid: egs/fma_genre/valid
  evaluate: egs/fma_genre/valid
  generate: egs/fma_genre/valid
"""
    config_path = config_dir / "fma_genre.yaml"
    with open(config_path, "w") as f:
        f.write(yaml_content)
    print(f"  ✓ {config_path}")

    readme = out_root / "FINETUNE_README.txt"
    with open(readme, "w") as f:
        f.write(
            "MusicGen Fine-tuning — Next Steps\n"
            "=================================\n\n"
            "1. Install AudioCraft (GPU machine / Colab T4):\n"
            "   pip install audiocraft\n"
            "   git clone https://github.com/facebookresearch/audiocraft.git\n"
            "   cd audiocraft\n\n"
            "2. Copy musicgen_data/ and config/ into audiocraft repo, then create manifests:\n"
            "   python -m audiocraft.data.audio_dataset ../musicgen_data/train egs/fma_genre/train/data.jsonl.gz\n"
            "   python -m audiocraft.data.audio_dataset ../musicgen_data/valid egs/fma_genre/valid/data.jsonl.gz\n\n"
            "3. Fine-tune MusicGen-small:\n"
            "   dora run solver=musicgen/musicgen_base_32khz \\\n"
            "     model/lm/model_scale=small \\\n"
            "     continue_from=//pretrained/facebook/musicgen-small \\\n"
            "     dset=audio/fma_genre \\\n"
            "     optim.lr=1e-5 \\\n"
            "     dataset.batch_size=4 \\\n"
            "     optim.epochs=25\n\n"
            "4. Generate samples:\n"
            "   python ../musicgen_generate.py --checkpoint <dora_sig> --prompt \"hip hop music with beats and rhythm\"\n\n"
            "Expected runtime: ~4-6 hours on Colab T4 for 800 tracks.\n"
        )
    print(f"  ✓ {readme}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FMA data for MusicGen fine-tuning")
    parser.add_argument(
        "--genres",
        nargs="+",
        default=DEFAULT_GENRES,
        help=f"Genres to include (default: {DEFAULT_GENRES})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Tracks per genre (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default="musicgen_data",
        help="Output directory (default: musicgen_data/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-split",
        default="train",
        help="HuggingFace split to draw from (default: train)",
    )
    args = parser.parse_args()

    stats = download_and_prepare(
        genres=args.genres,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        seed=args.seed,
        hf_split=args.hf_split,
    )

    total_train = sum(stats["train"].values())
    total_valid = sum(stats["valid"].values())
    print("\n" + "=" * 60)
    print("FMA DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Genres:      {args.genres}")
    print(f"  Train pairs: {total_train}")
    print(f"  Valid pairs: {total_valid}")
    print(f"  Output:      {args.output_dir}/")
    print("\nNext: run musicgen_finetune.py or follow musicgen_data/FINETUNE_README.txt")


if __name__ == "__main__":
    main()

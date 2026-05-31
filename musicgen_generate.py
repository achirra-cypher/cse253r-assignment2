#!/usr/bin/env python3
"""
musicgen_generate.py — Generate audio with MusicGen (pretrained or fine-tuned).

Uses HuggingFace transformers (lighter than full AudioCraft for inference).
Works on CPU (slow) or GPU. Produces the main deliverable continuous_conditioned.mp3.

Usage:
  pip install transformers scipy torchaudio accelerate
  python musicgen_generate.py --prompt "hip hop music with beats and rhythm"
  python musicgen_generate.py --prompt "acoustic folk music with guitar" --output folk_sample.mp3
  python musicgen_generate.py --all-genres                    # one sample per target genre
  python musicgen_generate.py --checkpoint ./finetuned_ckpt # after fine-tuning

Outputs:
  generated_audio/           — individual MP3/WAV samples
  continuous_conditioned.mp3 — main deliverable (best sample)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

GENRE_PROMPTS = {
    "Hip-Hop": "hip hop music with beats and rhythm",
    "Folk": "acoustic folk music with guitar",
    "Electronic": "electronic music with synthesizers",
    "Rock": "energetic rock music with electric guitar",
}

DEFAULT_MODEL = "facebook/musicgen-small"
SAMPLE_RATE = 32000


def _save_audio(wav: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save float32 mono audio to MP3 (via scipy) or WAV."""
    import scipy.io.wavfile as wavfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(wav, -1.0, 1.0)
    if path.suffix.lower() == ".mp3":
        try:
            from scipy.io import wavfile as _wf
            tmp_wav = path.with_suffix(".wav")
            _wf.write(str(tmp_wav), sample_rate, (audio * 32767).astype(np.int16))
            # Try ffmpeg for mp3
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_wav), "-q:a", "2", str(path)],
                check=True,
                capture_output=True,
            )
            tmp_wav.unlink(missing_ok=True)
        except Exception:
            # Fallback: save as WAV with .mp3 extension replaced
            wav_path = path.with_suffix(".wav")
            wavfile.write(str(wav_path), sample_rate, (audio * 32767).astype(np.int16))
            print(f"  (ffmpeg unavailable — saved {wav_path} instead of MP3)")
    else:
        wavfile.write(str(path), sample_rate, (audio * 32767).astype(np.int16))


def load_model(model_id: str, checkpoint: str | None, device: str):
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import torch

    if checkpoint and Path(checkpoint).exists():
        print(f"  Loading fine-tuned checkpoint: {checkpoint}")
        model = MusicgenForConditionalGeneration.from_pretrained(checkpoint)
        processor = AutoProcessor.from_pretrained(checkpoint)
    else:
        print(f"  Loading pretrained: {model_id}")
        model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

    model = model.to(device)
    model.eval()
    return model, processor


def generate_one(
    model,
    processor,
    prompt: str,
    duration_sec: float,
    device: str,
    seed: int | None = None,
) -> np.ndarray:
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # MusicGen generates ~50 tokens/sec at 32kHz with default codec
    max_new_tokens = int(duration_sec * 50)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    wav = audio_values[0, 0].cpu().numpy()
    return wav


def main():
    parser = argparse.ArgumentParser(description="Generate audio with MusicGen")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned checkpoint")
    parser.add_argument("--prompt", default=None, help="Text prompt for generation")
    parser.add_argument("--all-genres", action="store_true", help="Generate for all 4 target genres")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--output", default="continuous_conditioned.mp3", help="Main output file")
    parser.add_argument("--out-dir", default="generated_audio", help="Directory for all samples")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("MusicGen Audio Generation")
    print("=" * 60)
    print(f"  Device: {device}")

    model, processor = load_model(args.model, args.checkpoint, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    if args.all_genres:
        prompts = list(GENRE_PROMPTS.items())
    elif args.prompt:
        prompts = [("custom", args.prompt)]
    else:
        prompts = [("Hip-Hop", GENRE_PROMPTS["Hip-Hop"])]

    best_wav = None
    best_path = None

    for i, (genre, prompt) in enumerate(prompts):
        print(f"\n  Generating [{genre}]: '{prompt}'")
        wav = generate_one(
            model, processor, prompt, args.duration, device, seed=args.seed + i
        )
        safe_name = genre.replace(" ", "_").lower()
        out_path = out_dir / f"{safe_name}_generated.mp3"
        _save_audio(wav, str(out_path))
        print(f"  ✓ {out_path} ({len(wav) / SAMPLE_RATE:.1f}s)")

        manifest.append({"genre": genre, "prompt": prompt, "path": str(out_path)})
        if i == 0 or genre == "Hip-Hop":
            best_wav = wav
            best_path = out_path

    # Main deliverable
    if best_wav is not None:
        _save_audio(best_wav, args.output)
        print(f"\n  ★ Main deliverable: {args.output}")

    manifest_path = out_dir / "generation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "model": args.checkpoint or args.model,
                "device": device,
                "duration_sec": args.duration,
                "samples": manifest,
                "main_deliverable": args.output,
            },
            f,
            indent=2,
        )
    print(f"  ✓ {manifest_path}")


if __name__ == "__main__":
    main()

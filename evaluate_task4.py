#!/usr/bin/env python3
"""
evaluate_task4.py — Task 4 evaluation: genre consistency & baseline comparison (Day 4).

Compares pretrained vs fine-tuned MusicGen using:
  - Genre classifier accuracy (train simple classifier on FMA MFCC features)
  - Side-by-side generation manifest for qualitative listening

Optional (if fad-torch installed):
  - Frechet Audio Distance vs real FMA samples

Prerequisites:
  - generated_audio/ from musicgen_generate.py (pretrained baseline)
  - finetuned_musicgen/ + generated samples from fine-tuned model (optional)
  - musicgen_data/ from prepare_fma_data.py (for classifier training)

Usage:
  python evaluate_task4.py
  python evaluate_task4.py --skip-classifier   # if no FMA data downloaded yet

Produces:
  evaluation_task4.json
  eval_task4_genre_accuracy.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET_GENRES = ["hip-hop", "folk", "electronic", "rock"]
GENRE_LABELS = ["Hip-Hop", "Folk", "Electronic", "Rock"]

GENRE_PROMPTS = {
    "Hip-Hop": "hip hop music with beats and rhythm",
    "Folk": "acoustic folk music with guitar",
    "Electronic": "electronic music with synthesizers",
    "Rock": "energetic rock music with electric guitar",
}


def extract_mfcc(wav: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=wav.astype(np.float32), sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def train_genre_classifier(data_dir: str = "musicgen_data", max_per_genre: int = 100):
    """Train a simple logistic regression genre classifier on FMA MFCC features."""
    import soundfile as sf
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    X, y = [], []
    data_root = Path(data_dir) / "train"
    if not data_root.exists():
        raise FileNotFoundError(f"{data_root} not found. Run prepare_fma_data.py first.")

    for genre_dir in sorted(data_root.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre_name = genre_dir.name.replace("_", "-")
        if genre_name.lower() not in [g.lower() for g in GENRE_LABELS]:
            # Map folder names back
            genre_name = genre_dir.name.replace("_", " ")
        count = 0
        for wav_path in sorted(genre_dir.glob("*.wav")):
            if count >= max_per_genre:
                break
            audio, sr = sf.read(str(wav_path))
            X.append(extract_mfcc(audio, sr))
            y.append(genre_dir.name.replace("_", " "))
            count += 1

    if len(X) < 20:
        raise ValueError("Not enough training data for classifier.")

    X = np.stack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X, y_enc)

    train_acc = float(clf.score(X, y_enc))
    print(f"  Genre classifier train accuracy: {train_acc:.3f}")
    return clf, le


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from WAV/MP3 (librosa handles MP3 when soundfile cannot)."""
    import librosa
    import soundfile as sf

    try:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32), int(sr)
    except Exception:
        audio, sr = librosa.load(path, sr=None, mono=True)
        return audio.astype(np.float32), int(sr)


def _resolve_sample_path(gen_dir: Path, raw_path: str) -> Path | None:
    """Resolve manifest or glob path relative to gen_dir or cwd."""
    candidates = [
        Path(raw_path),
        gen_dir / Path(raw_path).name,
        gen_dir / raw_path,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _discover_generated_samples(gen_dir: str | Path) -> dict[str, Path]:
    """Map genre label (e.g. 'Hip-Hop') -> audio file path."""
    gen_dir = Path(gen_dir)
    found: dict[str, Path] = {}
    if not gen_dir.exists():
        return found

    manifest_path = gen_dir / "generation_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        for sample in data.get("samples", []):
            genre = sample.get("genre")
            raw = sample.get("path")
            if not genre or not raw:
                continue
            resolved = _resolve_sample_path(gen_dir, raw)
            if resolved:
                found[genre] = resolved

    for genre in GENRE_PROMPTS:
        if genre in found:
            continue
        slug = genre.replace(" ", "-").replace("_", "-").lower()
        for pattern in (f"{slug}_generated.*", f"{slug}*"):
            matches = sorted(gen_dir.glob(pattern))
            matches = [p for p in matches if p.suffix.lower() in (".mp3", ".wav")]
            if matches:
                found[genre] = matches[0]
                break

    return found


def _normalize_genre_label(name: str) -> str:
    return name.lower().replace("_", "-").replace(" ", "-")


def classify_generated(clf, le, audio_path: str) -> tuple[str, float]:
    audio, sr = _load_audio(audio_path)
    feat = extract_mfcc(audio, sr).reshape(1, -1)
    pred = clf.predict(feat)[0]
    proba = clf.predict_proba(feat)[0].max()
    return le.classes_[pred], float(proba)


def evaluate_genre_consistency(
    clf, le, generated_dir: str, expected_genres: dict[str, str]
) -> dict:
    """Check if classifier agrees with target genre for each generated sample."""
    results = []
    gen_dir = Path(generated_dir)
    if not gen_dir.exists():
        return {"error": f"{generated_dir} not found", "samples": []}

    discovered = _discover_generated_samples(gen_dir)
    if not discovered:
        return {"error": f"no audio files in {generated_dir}", "samples": []}

    for genre, prompt in expected_genres.items():
        path_obj = discovered.get(genre)
        if not path_obj:
            continue

        path = str(path_obj)
        pred_genre, confidence = classify_generated(clf, le, path)
        match = _normalize_genre_label(pred_genre) == _normalize_genre_label(genre)
        results.append(
            {
                "target_genre": genre,
                "prompt": prompt,
                "file": path,
                "predicted_genre": pred_genre,
                "confidence": confidence,
                "match": match,
            }
        )

    accuracy = (
        sum(r["match"] for r in results) / len(results) if results else 0.0
    )
    return {"genre_accuracy": accuracy, "samples": results}


def compare_pretrained_vs_finetuned(
    pretrained_dir: str = "generated_audio",
    finetuned_dir: str = "generated_audio_finetuned",
) -> dict:
    """Build comparison table for qualitative evaluation."""
    pre_map = _discover_generated_samples(pretrained_dir)
    ft_map = _discover_generated_samples(finetuned_dir)
    comparison = []
    for genre in GENRE_PROMPTS:
        comparison.append(
            {
                "genre": genre,
                "prompt": GENRE_PROMPTS[genre],
                "pretrained_file": str(pre_map[genre]) if genre in pre_map else None,
                "finetuned_file": str(ft_map[genre]) if genre in ft_map else None,
            }
        )
    return comparison


def plot_genre_accuracy(pretrained_results: dict, finetuned_results: dict | None):
    pre_samples = pretrained_results.get("samples", [])
    ft_samples = (finetuned_results or {}).get("samples", [])
    # Prefer fine-tuned sample list when pretrained baseline was not generated
    anchor = ft_samples or pre_samples
    labels = [r["target_genre"] for r in anchor]
    if not labels:
        return

    pre_by_genre = {r["target_genre"]: r for r in pre_samples}
    ft_by_genre = {r["target_genre"]: r for r in ft_samples}
    pre_match = [1 if pre_by_genre.get(g, {}).get("match") else 0 for g in labels]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    if pre_samples:
        ax.bar(x - width / 2 if ft_samples else x, pre_match, width, label="Pretrained", color="#3498db")

    if ft_samples:
        ft_match = [1 if ft_by_genre[g]["match"] else 0 for g in labels]
        ax.bar(x + width / 2 if pre_samples else x, ft_match, width, label="Fine-tuned", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Genre match (1=yes)")
    ax.set_title("Genre Consistency: Classifier Agreement with Target Genre", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("eval_task4_genre_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ eval_task4_genre_accuracy.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="musicgen_data")
    parser.add_argument("--pretrained-dir", default="generated_audio")
    parser.add_argument("--finetuned-dir", default="generated_audio_finetuned")
    parser.add_argument("--skip-classifier", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 4 Evaluation")
    print("=" * 60)

    results = {"comparison": compare_pretrained_vs_finetuned(args.pretrained_dir, args.finetuned_dir)}

    if not args.skip_classifier:
        try:
            clf, le = train_genre_classifier(args.data_dir)
            pre_eval = evaluate_genre_consistency(clf, le, args.pretrained_dir, GENRE_PROMPTS)
            results["pretrained"] = pre_eval
            print(f"  Pretrained genre accuracy: {pre_eval.get('genre_accuracy', 0):.3f}")

            ft_dir = Path(args.finetuned_dir)
            ft_map = _discover_generated_samples(ft_dir)
            if ft_map:
                ft_eval = evaluate_genre_consistency(clf, le, args.finetuned_dir, GENRE_PROMPTS)
                results["finetuned"] = ft_eval
                print(f"  Fine-tuned genre accuracy:  {ft_eval.get('genre_accuracy', 0):.3f}")
                plot_genre_accuracy(pre_eval, ft_eval)
            else:
                print(f"  (No fine-tuned audio in {args.finetuned_dir} — skipping)")
                if pre_eval.get("samples"):
                    plot_genre_accuracy(pre_eval, None)
                else:
                    print("  (No generated audio found — run musicgen_generate.py first)")

        except FileNotFoundError as e:
            print(f"  Skipping classifier: {e}")
            results["classifier_error"] = str(e)
    else:
        print("  Classifier skipped (--skip-classifier)")

    with open("evaluation_task4.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ evaluation_task4.json")

    print("\n" + "=" * 60)
    print("QUALITATIVE COMPARISON (listen to these pairs)")
    print("=" * 60)
    for row in results["comparison"]:
        print(f"  {row['genre']}:")
        print(f"    Prompt:     {row['prompt']}")
        print(f"    Pretrained: {row['pretrained_file'] or '(not generated yet)'}")
        print(f"    Fine-tuned: {row['finetuned_file'] or '(not generated yet)'}")


if __name__ == "__main__":
    main()

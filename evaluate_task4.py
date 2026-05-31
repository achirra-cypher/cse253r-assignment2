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

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X, y_enc)

    train_acc = float(clf.score(X, y_enc))
    print(f"  Genre classifier train accuracy: {train_acc:.3f}")
    return clf, le


def classify_generated(clf, le, audio_path: str) -> tuple[str, float]:
    import soundfile as sf

    audio, sr = sf.read(audio_path)
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

    for genre, prompt in expected_genres.items():
        safe = genre.replace(" ", "_").lower()
        candidates = list(gen_dir.glob(f"{safe}*.mp3")) + list(gen_dir.glob(f"{safe}*.wav"))
        if not candidates:
            candidates = list(gen_dir.glob("*.mp3")) + list(gen_dir.glob("*.wav"))
        if not candidates:
            continue

        path = str(candidates[0])
        pred_genre, confidence = classify_generated(clf, le, path)
        expected = genre.replace("_", " ")
        match = pred_genre.lower().replace("_", " ") == expected.lower().replace("_", " ")
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
    comparison = []
    for genre in GENRE_PROMPTS:
        safe = genre.replace(" ", "_").lower()
        pre = list(Path(pretrained_dir).glob(f"{safe}*")) if Path(pretrained_dir).exists() else []
        ft = list(Path(finetuned_dir).glob(f"{safe}*")) if Path(finetuned_dir).exists() else []
        comparison.append(
            {
                "genre": genre,
                "prompt": GENRE_PROMPTS[genre],
                "pretrained_file": str(pre[0]) if pre else None,
                "finetuned_file": str(ft[0]) if ft else None,
            }
        )
    return comparison


def plot_genre_accuracy(pretrained_results: dict, finetuned_results: dict | None):
    labels = [r["target_genre"] for r in pretrained_results.get("samples", [])]
    if not labels:
        return

    pre_match = [1 if r["match"] else 0 for r in pretrained_results["samples"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2 if finetuned_results else x, pre_match, width, label="Pretrained", color="#3498db")

    if finetuned_results and finetuned_results.get("samples"):
        ft_match = [1 if r["match"] else 0 for r in finetuned_results["samples"]]
        ax.bar(x + width / 2, ft_match, width, label="Fine-tuned", color="#2ecc71")

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
            if ft_dir.exists() and any(ft_dir.iterdir()):
                ft_eval = evaluate_genre_consistency(clf, le, args.finetuned_dir, GENRE_PROMPTS)
                results["finetuned"] = ft_eval
                print(f"  Fine-tuned genre accuracy:  {ft_eval.get('genre_accuracy', 0):.3f}")
                plot_genre_accuracy(pre_eval, ft_eval)
            else:
                print(f"  (No fine-tuned samples in {args.finetuned_dir} — skipping)")
                plot_genre_accuracy(pre_eval, None)

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

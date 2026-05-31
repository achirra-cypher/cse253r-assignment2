#!/usr/bin/env python3
"""
evaluate_task1.py — Task 1 evaluation metrics (Day 4).

Computes:
  - Test perplexity (Markov vs LSTM)
  - Pitch-class KL divergence (generated vs real Bach)
  - Melodic interval L1 distance
  - Voice range violation rate
  - Parallel 5ths / octaves violation rate

Produces:
  - evaluation_task1.json
  - eval_pitch_kl.png
  - eval_intervals.png
  - eval_summary.png

Usage:
  python evaluate_task1.py
  python evaluate_task1.py --n-samples 50   # fewer generated samples (faster)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models import BigramMarkovChain, ChoraleLSTM, generate_lstm

VOICE_RANGES = {
    0: (57, 81),  # Soprano
    1: (53, 74),  # Alto
    2: (48, 69),  # Tenor
    3: (36, 64),  # Bass
}
VOICE_NAMES = ["Soprano", "Alto", "Tenor", "Bass"]


def load_data():
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    with open("vocab.json") as f:
        vdata = json.load(f)
    idx2token = {v: int(k) for k, v in vdata["token2idx"].items()}
    vocab_size = vdata["vocab_size"]
    return X_train, X_test, idx2token, vocab_size


def tokens_to_midi(tokens: np.ndarray, idx2token: dict) -> np.ndarray:
    """(T, 4) token indices → MIDI pitches."""
    out = np.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for v in range(4):
            out[t, v] = idx2token.get(int(tokens[t, v]), 0)
    return out


def pitch_class_distribution(rolls: list[np.ndarray]) -> np.ndarray:
    """12-dim pitch class histogram (ignoring rests)."""
    counts = np.zeros(12, dtype=np.float64)
    for roll in rolls:
        for v in range(4):
            pitches = roll[:, v]
            pitches = pitches[pitches > 0]
            for p in pitches:
                counts[int(p) % 12] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(12) / 12
    return counts / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def melodic_intervals(roll: np.ndarray, voice: int) -> list[int]:
    """Signed semitone intervals for one voice (skip rests / unisons)."""
    pitches = roll[:, voice]
    intervals = []
    prev = None
    for p in pitches:
        p = int(p)
        if p == 0:
            prev = None
            continue
        if prev is not None and p != prev:
            intervals.append(p - prev)
        prev = p
    return intervals


def interval_histogram(rolls: list[np.ndarray], voice: int | None = None) -> np.ndarray:
    """Histogram over interval bins [-12, ..., 12] (25 bins)."""
    bins = np.arange(-12, 13)
    hist = np.zeros(len(bins), dtype=np.float64)
    for roll in rolls:
        voices = range(4) if voice is None else [voice]
        for v in voices:
            for iv in melodic_intervals(roll, v):
                if -12 <= iv <= 12:
                    hist[iv + 12] += 1
    total = hist.sum()
    if total == 0:
        return hist
    return hist / total


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def voice_range_violation_rate(roll: np.ndarray) -> float:
    """Fraction of non-rest notes outside historical SATB ranges."""
    violations = 0
    total = 0
    for v in range(4):
        lo, hi = VOICE_RANGES[v]
        pitches = roll[:, v]
        for p in pitches:
            p = int(p)
            if p == 0:
                continue
            total += 1
            if p < lo or p > hi:
                violations += 1
    return violations / max(total, 1)


def _interval_class(p1: int, p2: int) -> int:
    """Semitone interval modulo octave (0=unison, 7=fifth)."""
    return abs(p2 - p1) % 12


def parallel_fifths_octaves_rate(roll: np.ndarray) -> float:
    """
    Count voice-leading violations: two voices moving in parallel
    into a perfect 5th or octave between outer voices.
    """
    T = roll.shape[0]
    violations = 0
    comparisons = 0

    for t in range(T - 1):
        cur = roll[t]
        nxt = roll[t + 1]
        active_voices = [v for v in range(4) if cur[v] > 0 and nxt[v] > 0]
        if len(active_voices) < 2:
            continue

        for i, v1 in enumerate(active_voices):
            for v2 in active_voices[i + 1 :]:
                p1, p2 = int(cur[v1]), int(cur[v2])
                n1, n2 = int(nxt[v1]), int(nxt[v2])
                if p1 == n1 and p2 == n2:
                    continue  # sustained

                d1, d2 = n1 - p1, n2 - p2
                if d1 == 0 or d2 == 0:
                    continue
                if np.sign(d1) != np.sign(d2):
                    continue  # contrary / oblique motion

                ic_cur = _interval_class(p1, p2)
                ic_nxt = _interval_class(n1, n2)
                comparisons += 1
                if ic_cur in (0, 7) and ic_nxt in (0, 7):
                    violations += 1

    return violations / max(comparisons, 1)


def lstm_test_perplexity(model, X_test, vocab_size, device="cpu") -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="mean")
    inp = torch.from_numpy(X_test[:, :-1, :]).long().to(device)
    tgt = torch.from_numpy(X_test[:, 1:, :]).long().to(device)
    with torch.no_grad():
        logits = model(inp)
        loss = sum(
            criterion(
                logits[:, :, v, :].reshape(-1, vocab_size),
                tgt[:, :, v].reshape(-1),
            ).item()
            for v in range(4)
        )
    return float(np.exp(loss / 4.0))


def generate_samples(mc, model, X_train, n_samples, length, device):
    markov_rolls, lstm_rolls = [], []
    for i in range(n_samples):
        seed = X_train[i % len(X_train), 0, :]
        m_tok = mc.generate(length=length, temperature=0.8, seed_token=seed)
        l_tok = generate_lstm(
            model, None, length=length, temperature=0.9, seed=seed, device=device
        )
        markov_rolls.append(m_tok)
        lstm_rolls.append(l_tok)
    return markov_rolls, lstm_rolls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("Task 1 Evaluation")
    print("=" * 60)

    X_train, X_test, idx2token, vocab_size = load_data()

    # Fit / load models
    mc = BigramMarkovChain(vocab_size=vocab_size)
    mc.fit(X_train)
    markov_ppl = mc.perplexity(X_test)

    model = ChoraleLSTM(vocab_size=vocab_size)
    model.load_state_dict(torch.load("lstm_checkpoint.pt", map_location=args.device))
    model = model.to(args.device)
    lstm_ppl = lstm_test_perplexity(model, X_test, vocab_size, args.device)

    print(f"  Markov test perplexity: {markov_ppl:.3f}")
    print(f"  LSTM test perplexity:   {lstm_ppl:.3f}")

    # Real Bach rolls from test set
    real_rolls = [tokens_to_midi(X_test[i], idx2token) for i in range(min(args.n_samples, len(X_test)))]
    markov_tok, lstm_tok = generate_samples(
        mc, model, X_train, args.n_samples, args.gen_length, args.device
    )
    markov_rolls = [tokens_to_midi(t, idx2token) for t in markov_tok]
    lstm_rolls = [tokens_to_midi(t, idx2token) for t in lstm_tok]

    # Pitch-class KL divergence
    p_real = pitch_class_distribution(real_rolls)
    p_markov = pitch_class_distribution(markov_rolls)
    p_lstm = pitch_class_distribution(lstm_rolls)
    kl_markov = kl_divergence(p_real, p_markov)
    kl_lstm = kl_divergence(p_real, p_lstm)

    # Interval L1
    h_real = interval_histogram(real_rolls)
    h_markov = interval_histogram(markov_rolls)
    h_lstm = interval_histogram(lstm_rolls)
    l1_markov = l1_distance(h_real, h_markov)
    l1_lstm = l1_distance(h_real, h_lstm)

    # Voice range & parallel 5ths/octaves
    def avg_metric(rolls, fn):
        return float(np.mean([fn(r) for r in rolls]))

    range_markov = avg_metric(markov_rolls, voice_range_violation_rate)
    range_lstm = avg_metric(lstm_rolls, voice_range_violation_rate)
    range_real = avg_metric(real_rolls, voice_range_violation_rate)
    parallel_markov = avg_metric(markov_rolls, parallel_fifths_octaves_rate)
    parallel_lstm = avg_metric(lstm_rolls, parallel_fifths_octaves_rate)
    parallel_real = avg_metric(real_rolls, parallel_fifths_octaves_rate)

    results = {
        "perplexity": {"markov": markov_ppl, "lstm": lstm_ppl},
        "pitch_class_kl_to_real": {"markov": kl_markov, "lstm": kl_lstm},
        "interval_l1_to_real": {"markov": l1_markov, "lstm": l1_lstm},
        "voice_range_violation_rate": {
            "real": range_real,
            "markov": range_markov,
            "lstm": range_lstm,
        },
        "parallel_fifths_octaves_rate": {
            "real": parallel_real,
            "markov": parallel_markov,
            "lstm": parallel_lstm,
        },
        "n_generated_samples": args.n_samples,
        "gen_length": args.gen_length,
    }

    with open("evaluation_task1.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ evaluation_task1.json")

    # ── Plots ──
    pc_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, dist, title, color in zip(
        axes,
        [p_real, p_markov, p_lstm],
        ["Real Bach", "Markov", "LSTM"],
        ["#2c3e50", "#e74c3c", "#2ecc71"],
    ):
        ax.bar(pc_labels, dist, color=color, alpha=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Pitch class")
        ax.set_ylabel("Frequency")
    fig.suptitle("Pitch-Class Distribution (lower KL to Real = better)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("eval_pitch_kl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ eval_pitch_kl.png")

    iv_labels = list(range(-12, 13))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(iv_labels, h_real, "o-", label="Real", linewidth=2)
    ax.plot(iv_labels, h_markov, "s--", label="Markov", alpha=0.8)
    ax.plot(iv_labels, h_lstm, "^--", label="LSTM", alpha=0.8)
    ax.set_xlabel("Melodic interval (semitones)")
    ax.set_ylabel("Normalized frequency")
    ax.set_title("Melodic Interval Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eval_intervals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ eval_intervals.png")

    # Summary bar chart
    metrics = ["KL↓", "Interval L1↓", "Range viol.↓", "Parallel 5/8↓"]
    markov_vals = [kl_markov, l1_markov, range_markov, parallel_markov]
    lstm_vals = [kl_lstm, l1_lstm, range_lstm, parallel_lstm]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, markov_vals, width, label="Markov", color="#e74c3c")
    ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Task 1 Evaluation Summary (lower is better)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("eval_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ eval_summary.png")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<28} {'Markov':>10} {'LSTM':>10}")
    print(f"  {'Perplexity':<28} {markov_ppl:>10.3f} {lstm_ppl:>10.3f}")
    print(f"  {'Pitch-class KL':<28} {kl_markov:>10.4f} {kl_lstm:>10.4f}")
    print(f"  {'Interval L1':<28} {l1_markov:>10.4f} {l1_lstm:>10.4f}")
    print(f"  {'Voice range violations':<28} {range_markov:>10.4f} {range_lstm:>10.4f}")
    print(f"  {'Parallel 5ths/octaves':<28} {parallel_markov:>10.4f} {parallel_lstm:>10.4f}")
    print(f"\n  (Real Bach reference — range: {range_real:.4f}, parallel: {parallel_real:.4f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_data.py — Preprocess JSB Chorales from music21 corpus.

Produces:
  jsb_chorales.npy   — 368 chorales as (T,4) MIDI pitch arrays
  jsb_paths.npy      — Corresponding music21 corpus paths
  vocab.json         — 47-token vocabulary + mappings
  split.json         — Train/val/test chorale indices
  X_train.npy        — Training windows (N, 64, 4)
  X_val.npy          — Validation windows
  X_test.npy         — Test windows
  test_chorale.mid   — Sample MIDI to verify pipeline
"""

import numpy as np
import json
import os
import sys

# ── Step 1: Load chorales from music21 ───────────────────────────────────────

print("=" * 60)
print("Step 1: Loading Bach Chorales from music21 corpus")
print("=" * 60)

import music21
from music21 import corpus

# Get all Bach chorale paths
bach_paths = corpus.getComposer('bach')
print(f"  Found {len(bach_paths)} Bach works in music21 corpus")

chorales = []
chorale_paths = []
skipped = 0

for i, path in enumerate(bach_paths):
    try:
        score = corpus.parse(path)
        parts = score.parts
        if len(parts) != 4:
            skipped += 1
            continue

        # Quantize to 16th notes and extract MIDI pitches
        # Each chorale → (T, 4) array: [Soprano, Alto, Tenor, Bass]
        max_len = 0
        voice_pitches = []

        for part in parts:
            pitches = []
            for element in part.flat.notesAndRests:
                # Duration in 16th notes
                dur_16ths = int(round(element.quarterLength * 4))
                dur_16ths = max(dur_16ths, 1)  # minimum 1 step

                if element.isRest:
                    pitches.extend([0] * dur_16ths)
                elif element.isNote:
                    pitches.extend([element.pitch.midi] * dur_16ths)
                elif element.isChord:
                    # Take highest note for soprano, lowest for bass, etc.
                    pitches.extend([element.pitches[-1].midi] * dur_16ths)

            voice_pitches.append(pitches)

        # Align all 4 voices to the same length
        min_len = min(len(v) for v in voice_pitches)
        if min_len < 16:  # skip very short chorales
            skipped += 1
            continue

        roll = np.zeros((min_len, 4), dtype=np.int32)
        for vi in range(4):
            roll[:, vi] = voice_pitches[vi][:min_len]

        chorales.append(roll)
        chorale_paths.append(str(path))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(bach_paths)} works, "
                  f"kept {len(chorales)} chorales so far...")

    except Exception as e:
        skipped += 1
        continue

print(f"\n  Result: {len(chorales)} four-part chorales loaded "
      f"({skipped} works skipped)")

# ── Step 2: Build vocabulary ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 2: Building vocabulary")
print("=" * 60)

all_pitches = set()
for c in chorales:
    all_pitches.update(c.flatten().tolist())
all_pitches.discard(0)  # 0 is rest/pad, handle separately

sorted_pitches = sorted(all_pitches)
vocab = [0] + sorted_pitches  # 0 = REST at index 0
token2idx = {p: i for i, p in enumerate(vocab)}

VOCAB_SIZE = len(vocab)
print(f"  Vocabulary size: {VOCAB_SIZE}")
print(f"  MIDI pitch range: {min(sorted_pitches)} - {max(sorted_pitches)}")
print(f"  Token 0 = REST/PAD")

# ── Step 3: Train/Val/Test split ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 3: Creating train/val/test split")
print("=" * 60)

n = len(chorales)
np.random.seed(42)
perm = np.random.permutation(n).tolist()

n_test = max(1, int(0.1 * n))
n_val = max(1, int(0.1 * n))
n_train = n - n_test - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

print(f"  Train: {len(train_idx)} chorales")
print(f"  Val:   {len(val_idx)} chorales")
print(f"  Test:  {len(test_idx)} chorales")

# ── Step 4: Create overlapping windows ───────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4: Creating sequence windows")
print("=" * 60)


def encode_roll(roll, token2idx):
    """(T,4) MIDI pitch → (T,4) token indices."""
    out = np.zeros_like(roll)
    for t in range(roll.shape[0]):
        for v in range(4):
            out[t, v] = token2idx.get(int(roll[t, v]), 0)
    return out


def make_sequences(chorales, token2idx, seq_len=64, stride=16, indices=None):
    """Slice chorales into overlapping windows of shape (seq_len, 4)."""
    seqs = []
    idx_list = indices if indices is not None else range(len(chorales))
    for ci in idx_list:
        enc = encode_roll(chorales[ci], token2idx)
        T = len(enc)
        for start in range(0, T - seq_len, stride):
            seqs.append(enc[start:start + seq_len])
    if len(seqs) == 0:
        return np.zeros((0, seq_len, 4), dtype=np.int32)
    return np.stack(seqs)


X_train = make_sequences(chorales, token2idx, seq_len=64, stride=16,
                         indices=train_idx)
X_val = make_sequences(chorales, token2idx, seq_len=64, stride=32,
                       indices=val_idx)
X_test = make_sequences(chorales, token2idx, seq_len=64, stride=32,
                        indices=test_idx)

print(f"  X_train: {X_train.shape}  ({X_train.shape[0]} windows)")
print(f"  X_val:   {X_val.shape}  ({X_val.shape[0]} windows)")
print(f"  X_test:  {X_test.shape}  ({X_test.shape[0]} windows)")

# ── Step 5: Save everything ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 5: Saving preprocessed data")
print("=" * 60)

# Save chorales as object array (variable-length arrays)
np.save('jsb_chorales.npy', np.array(chorales, dtype=object), allow_pickle=True)
print(f"  ✓ jsb_chorales.npy ({len(chorales)} chorales)")

# Save paths
np.save('jsb_paths.npy', np.array(chorale_paths, dtype=object), allow_pickle=True)
print(f"  ✓ jsb_paths.npy")

# Save vocab
vocab_data = {
    'vocab': vocab,
    'token2idx': {str(k): v for k, v in token2idx.items()},
    'vocab_size': VOCAB_SIZE,
}
with open('vocab.json', 'w') as f:
    json.dump(vocab_data, f, indent=2)
print(f"  ✓ vocab.json (vocab_size={VOCAB_SIZE})")

# Save split
split_data = {
    'train': train_idx,
    'val': val_idx,
    'test': test_idx,
}
with open('split.json', 'w') as f:
    json.dump(split_data, f, indent=2)
print(f"  ✓ split.json")

# Save sequence windows
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
print(f"  ✓ X_train.npy {X_train.shape}")
print(f"  ✓ X_val.npy   {X_val.shape}")
print(f"  ✓ X_test.npy  {X_test.shape}")

# ── Step 6: Verify with MIDI export ─────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 6: Verifying pipeline with MIDI export")
print("=" * 60)

import pretty_midi

VOICE_NAMES = ['Soprano', 'Alto', 'Tenor', 'Bass']


def roll_to_midi(roll, out_path, bpm=100):
    """(T,4) MIDI pitch array → .mid file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    sps = 60.0 / bpm / 4  # seconds per 16th-note step
    voice_programs = [0, 0, 0, 43]  # Piano x3 + Cello

    for vi in range(min(4, roll.shape[1])):
        inst = pretty_midi.Instrument(program=voice_programs[vi],
                                      name=VOICE_NAMES[vi])
        t = 0
        while t < len(roll):
            pitch = int(roll[t, vi])
            if pitch == 0:
                t += 1
                continue
            dur = 1
            while t + dur < len(roll) and int(roll[t + dur, vi]) == pitch:
                dur += 1
            inst.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch,
                start=t * sps, end=(t + dur) * sps))
            t += dur
        pm.instruments.append(inst)
    pm.write(out_path)


roll_to_midi(chorales[0], 'test_chorale.mid')
print(f"  ✓ test_chorale.mid exported ({len(chorales[0])} steps)")

# Also export a real chorale from training set for comparison
roll_to_midi(chorales[train_idx[0]], 'sample_chorale_real.mid')
print(f"  ✓ sample_chorale_real.mid exported")

# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"  Chorales: {len(chorales)}")
print(f"  Vocab:    {VOCAB_SIZE} tokens")
print(f"  Train:    {X_train.shape[0]} windows of {X_train.shape[1]}×{X_train.shape[2]}")
print(f"  Val:      {X_val.shape[0]} windows")
print(f"  Test:     {X_test.shape[0]} windows")
print(f"\nAll files saved to: {os.getcwd()}")

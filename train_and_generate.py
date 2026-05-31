#!/usr/bin/env python3
"""
train_and_generate.py — Train LSTM, run Markov baseline, generate MIDI outputs.

Produces:
  - markov_chorale.mid          — Generated from Bigram Markov Chain
  - symbolic_unconditioned.mid  — Generated from trained LSTM
  - lstm_checkpoint.pt          — Best LSTM model weights
  - training_history.json       — Loss/perplexity per epoch
  - training_curves.png         — Training curve plot
  - distribution_comparison.png — Pitch distribution comparison
"""

import numpy as np
import json
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from models import (
    BigramMarkovChain,
    ChoraleLSTM,
    train_lstm,
    generate_lstm,
    plot_training_curves,
    compare_distributions,
)

# ── Load data ────────────────────────────────────────────────────────────────

print("Loading preprocessed data...")
X_train = np.load('X_train.npy')
X_val   = np.load('X_val.npy')
X_test  = np.load('X_test.npy')

with open('vocab.json') as f:
    vdata = json.load(f)
vocab     = vdata['vocab']
token2idx = {int(k): v for k, v in vdata['token2idx'].items()}
idx2token = {v: int(k) for k, v in token2idx.items()}

VOCAB_SIZE = len(vocab)

print(f"  X_train: {X_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Vocab:   {VOCAB_SIZE} tokens")


def decode_tokens(tokens):
    """(T, 4) token indices → (T, 4) MIDI pitches."""
    out = np.zeros_like(tokens)
    for t in range(tokens.shape[0]):
        for v in range(4):
            out[t, v] = idx2token.get(int(tokens[t, v]), 0)
    return out


# ── Load MIDI exporter ──────────────────────────────────────────────────────

import pretty_midi

VOICE_NAMES = ['Soprano', 'Alto', 'Tenor', 'Bass']


def roll_to_midi(roll, out_path, bpm=100):
    """(T,4) MIDI pitch array → .mid file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    sps = 60.0 / bpm / 4
    voice_programs = [0, 0, 0, 43]

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Bigram Markov Chain Baseline
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 1: Bigram Markov Chain Baseline")
print("=" * 60)

mc = BigramMarkovChain(vocab_size=VOCAB_SIZE)
mc.fit(X_train)

markov_test_ppl = mc.perplexity(X_test)
print(f"  Test perplexity: {markov_test_ppl:.2f}")

# Generate a sample and export to MIDI
np.random.seed(123)
# Seed with a realistic first token from the training set
seed_token = X_train[0, 0, :]  # first timestep of first training chorale
markov_gen = mc.generate(length=128, temperature=0.8, seed_token=seed_token)
markov_gen_midi = decode_tokens(markov_gen)
roll_to_midi(markov_gen_midi, 'markov_chorale.mid')
print(f"  ✓ markov_chorale.mid exported (128 steps)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Train LSTM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 2: Training LSTM")
print("=" * 60)

device = 'cpu'
print(f"  Device: {device}")

model = ChoraleLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=64,
    hidden_dim=256,
    n_layers=2,
    dropout=0.3,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

history = train_lstm(
    model, X_train, X_val,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    device=device,
)

# Save checkpoint
torch.save(model.state_dict(), 'lstm_checkpoint.pt')
print(f"\n  ✓ lstm_checkpoint.pt saved")

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f"  ✓ training_history.json saved")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs_range, history['train_loss'], label='Train loss', marker='o', markersize=3)
axes[0].plot(epochs_range, history['val_loss'], label='Val loss', marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (sum CE over 4 voices)')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history['val_perplexity'], label='Val perplexity',
             color='tab:green', marker='^', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Perplexity')
axes[1].set_title('Validation Perplexity')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ training_curves.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Compute LSTM Test Perplexity
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 3: LSTM Test Perplexity")
print("=" * 60)

model.eval()
import torch.nn as nn
criterion = nn.CrossEntropyLoss(reduction='mean')

inp_test = torch.from_numpy(X_test[:, :-1, :]).long()
tgt_test = torch.from_numpy(X_test[:, 1:, :]).long()

with torch.no_grad():
    logits = model(inp_test)
    loss = 0.0
    for v in range(4):
        loss += criterion(
            logits[:, :, v, :].reshape(-1, VOCAB_SIZE),
            tgt_test[:, :, v].reshape(-1),
        ).item()
    lstm_test_ppl = float(np.exp(loss / 4.0))

print(f"  Markov test perplexity: {markov_test_ppl:.2f}")
print(f"  LSTM test perplexity:   {lstm_test_ppl:.2f}")
print(f"  Improvement:            {markov_test_ppl / lstm_test_ppl:.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Generate from LSTM and Export
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 4: LSTM Generation")
print("=" * 60)

# Generate with different temperatures
for temp in [0.7, 0.9, 1.0, 1.2]:
    seed = X_train[42, 0:1, :]  # seed from a real chorale
    gen = generate_lstm(model, decode_tokens, length=128, temperature=temp, seed=seed, device=device)
    roll_to_midi(gen, f'lstm_chorale_t{temp:.1f}.mid')
    print(f"  ✓ lstm_chorale_t{temp:.1f}.mid exported")

# The main deliverable: best temperature
best_seed = X_train[10, 0:1, :]
lstm_best = generate_lstm(model, decode_tokens, length=192, temperature=0.9, seed=best_seed, device=device)
roll_to_midi(lstm_best, 'symbolic_unconditioned.mid')
print(f"\n  ★ symbolic_unconditioned.mid exported (192 steps = ~48 beats)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Distribution Comparison
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STEP 5: Distribution Comparison")
print("=" * 60)

# Generate many samples for comparison
markov_samples = []
lstm_samples = []
for i in range(20):
    seed = X_train[i, 0:1, :]
    m_gen = mc.generate(length=64, temperature=0.8, seed_token=seed[0])
    l_gen = generate_lstm(model, None, length=64, temperature=0.9, seed=seed, device=device)
    markov_samples.append(m_gen)
    lstm_samples.append(l_gen)

markov_all = np.stack(markov_samples)
lstm_all = np.stack(lstm_samples)

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
bins = np.arange(VOCAB_SIZE + 1) - 0.5

for ax, data, title, color in zip(
    axes,
    [X_test, markov_all, lstm_all],
    ['Real Bach (test set)', 'Bigram Markov', 'LSTM (ours)'],
    ['#2c3e50', '#e74c3c', '#2ecc71'],
):
    ax.hist(data.flatten(), bins=bins, density=True, alpha=0.75,
            edgecolor='white', color=color, linewidth=0.5)
    ax.set_xlabel('Token index')
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-0.5, VOCAB_SIZE - 0.5)
    ax.grid(True, alpha=0.3)

plt.suptitle('Token Distribution: Real vs Generated', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ distribution_comparison.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BASELINE COMPLETE — Summary")
print("=" * 60)
print(f"  Markov test perplexity:  {markov_test_ppl:.2f}")
print(f"  LSTM test perplexity:    {lstm_test_ppl:.2f}")
print(f"  LSTM parameters:         {n_params:,}")
print(f"  Training epochs:         {len(history['train_loss'])}")
print(f"  Final val loss:          {history['val_loss'][-1]:.4f}")
print(f"  Final val perplexity:    {history['val_perplexity'][-1]:.2f}")
print()
print("  Generated files:")
print("    ★ symbolic_unconditioned.mid  (LSTM, main deliverable)")
print("    • markov_chorale.mid")
print("    • lstm_chorale_t0.7.mid ... t1.2.mid")
print("    • lstm_checkpoint.pt")
print("    • training_history.json")
print("    • training_curves.png")
print("    • distribution_comparison.png")

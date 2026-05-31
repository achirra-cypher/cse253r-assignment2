#!/usr/bin/env python3
"""Update workbook.ipynb to add Section 2 modeling cells and fix the header."""

import json

with open('workbook.ipynb', 'r') as f:
    nb = json.load(f)

# ── Fix notebook header (Task 1 + Task 4, not Task 2) ───────────────────────
nb['cells'][0]['source'] = [
    "# Music Generation — Symbolic & Continuous\n",
    "## CSE 253R Assignment 2\n",
    "### Task 1: Symbolic Unconditioned Generation | Task 4: Continuous Conditioned Generation\n",
    "\n",
    "---\n",
    "\n",
    "## Overview\n",
    "\n",
    "This notebook explores two music generation tasks:\n",
    "\n",
    "- **Task 1 (Symbolic Unconditioned):** Train a model to learn the distribution\n",
    "  p(Soprano, Alto, Tenor, Bass) on the **JSB Chorales** dataset and freely generate\n",
    "  new 4-part Bach-style chorales.\n",
    "- **Task 4 (Continuous Conditioned):** Fine-tune **MusicGen-small** (Meta/AudioCraft)\n",
    "  on the **FMA-small** dataset to generate genre-specific audio from text prompts.\n",
    "\n",
    "Both tasks share a common theme — learning to generate music — but differ in\n",
    "representation (symbolic MIDI vs continuous audio) and conditioning (none vs text).\n"
]

# ── Find the Section 2 placeholder cell and replace it ───────────────────────
# Look for the cell with "Section 2: Modeling"
sec2_idx = None
sec3_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Section 2: Modeling' in src:
        sec2_idx = i
    if 'Section 3:' in src and sec2_idx is not None and sec3_idx is None:
        sec3_idx = i

assert sec2_idx is not None, "Could not find Section 2 placeholder"
assert sec3_idx is not None, "Could not find Section 3"

print(f"Section 2 placeholder at cell index {sec2_idx}")
print(f"Section 3 starts at cell index {sec3_idx}")

# New cells to insert
new_cells = [
    # Section 2 header
    {
        "cell_type": "markdown",
        "id": "b2e6d5a1",
        "metadata": {},
        "source": [
            "---\n",
            "## Section 2: Modeling — Task 1 (Unconditioned)\n",
            "\n",
            "We implement two models for symbolic unconditioned generation:\n",
            "\n",
            "1. **Bigram Markov Chain** — A simple baseline that learns per-voice transition probabilities\n",
            "2. **LSTM** — A 2-layer LSTM with per-voice embeddings and output heads\n",
            "\n",
            "Both models learn to predict the next timestep given the current one. The LSTM captures\n",
            "longer-range dependencies through its hidden state, while the Markov chain is limited to\n",
            "single-step context.\n"
        ]
    },
    # 2.1 Markov header
    {
        "cell_type": "markdown",
        "id": "sec2_markov_header",
        "metadata": {},
        "source": [
            "### 2.1 Bigram Markov Chain Baseline\n",
            "\n",
            "For each voice independently, we build a 47×47 transition count matrix from training data,\n",
            "apply Laplace (add-1) smoothing, and normalize to get conditional probabilities:\n",
            "\n",
            "$$P(\\text{next token} \\mid \\text{current token}) = \\frac{\\text{count}(\\text{current} \\to \\text{next}) + 1}{\\sum_{j} [\\text{count}(\\text{current} \\to j) + 1]}$$\n",
            "\n",
            "Generation proceeds autoregressively by sampling from the learned distribution.\n"
        ]
    },
    # 2.1 Markov code
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_markov_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import TensorDataset, DataLoader\n",
            "from models import BigramMarkovChain, ChoraleLSTM, train_lstm, generate_lstm\n",
            "\n",
            "# ── Fit Bigram Markov Chain ────────────────────────────────────────────────────\n",
            "mc = BigramMarkovChain(vocab_size=VOCAB_SIZE)\n",
            "mc.fit(X_train)\n",
            "\n",
            "# Evaluate on test set\n",
            "markov_test_ppl = mc.perplexity(X_test)\n",
            "print(f\"Bigram Markov Chain — Test Perplexity: {markov_test_ppl:.2f}\")\n",
            "\n",
            "# Visualize transition matrix for each voice\n",
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n",
            "for vi, (ax, vname) in enumerate(zip(axes, VOICE_NAMES)):\n",
            "    im = ax.imshow(mc.transition_probs[vi], aspect='auto', cmap='hot')\n",
            "    ax.set_title(f'{vname}', fontsize=10)\n",
            "    ax.set_xlabel('Next token')\n",
            "    if vi == 0:\n",
            "        ax.set_ylabel('Current token')\n",
            "fig.colorbar(im, ax=axes, shrink=0.6, label='P(next | current)')\n",
            "fig.suptitle('Bigram Transition Probabilities (per voice)', fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('markov_transitions.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Generate a sample chorale from the Markov chain\n",
            "np.random.seed(123)\n",
            "seed_token = X_train[0, 0, :]  # realistic seed from training data\n",
            "markov_gen = mc.generate(length=128, temperature=0.8, seed_token=seed_token)\n",
            "markov_gen_midi = tokenizer.decode(markov_gen)\n",
            "roll_to_midi(markov_gen_midi, 'markov_chorale.mid')\n",
            "print(f\"\\n✓ Generated Markov chorale (128 steps) → markov_chorale.mid\")\n",
            "print(f\"  Token range: [{markov_gen.min()}, {markov_gen.max()}]\")\n",
            "print(f\"  Unique tokens used: {len(np.unique(markov_gen))}\")"
        ]
    },
    # 2.2 LSTM header
    {
        "cell_type": "markdown",
        "id": "sec2_lstm_header",
        "metadata": {},
        "source": [
            "### 2.2 LSTM Model Architecture\n",
            "\n",
            "Our LSTM architecture processes all 4 voices jointly at each timestep:\n",
            "\n",
            "```\n",
            "Input: (batch, T, 4) token indices\n",
            "  ↓\n",
            "4 × Embedding(47, 64) → concatenate → (batch, T, 256)\n",
            "  ↓\n",
            "LSTM(256 input, 256 hidden, 2 layers, dropout=0.3)\n",
            "  ↓\n",
            "4 × Linear(256, 47) → logits: (batch, T, 4, 47)\n",
            "```\n",
            "\n",
            "**Teacher forcing:** During training, the model receives the ground-truth tokens at time t\n",
            "and predicts tokens at time t+1. Loss is the sum of cross-entropy over all 4 voices.\n",
            "\n",
            "**Generation:** Autoregressive sampling — start from a seed token, feed predictions back\n",
            "as input for the next step. Temperature controls diversity.\n"
        ]
    },
    # 2.2 LSTM training code
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_lstm_train",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Train LSTM ─────────────────────────────────────────────────────────────────\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f\"Training device: {device}\")\n",
            "\n",
            "model = ChoraleLSTM(\n",
            "    vocab_size=VOCAB_SIZE,\n",
            "    embed_dim=64,\n",
            "    hidden_dim=256,\n",
            "    n_layers=2,\n",
            "    dropout=0.3,\n",
            ")\n",
            "\n",
            "n_params = sum(p.numel() for p in model.parameters())\n",
            "print(f\"Model parameters: {n_params:,}\")\n",
            "\n",
            "# If pre-trained checkpoint exists, load it instead of re-training\n",
            "import os\n",
            "if os.path.exists('lstm_checkpoint.pt'):\n",
            "    model.load_state_dict(torch.load('lstm_checkpoint.pt', map_location=device))\n",
            "    model = model.to(device)\n",
            "    print(\"\\n✓ Loaded pre-trained checkpoint: lstm_checkpoint.pt\")\n",
            "    \n",
            "    # Load saved history if available\n",
            "    if os.path.exists('training_history.json'):\n",
            "        with open('training_history.json') as f:\n",
            "            history = json.load(f)\n",
            "        print(f\"  Epochs trained: {len(history['train_loss'])}\")\n",
            "        print(f\"  Best val loss:  {min(history['val_loss']):.4f}\")\n",
            "        print(f\"  Best val ppl:   {min(history['val_perplexity']):.2f}\")\n",
            "    else:\n",
            "        history = None\n",
            "else:\n",
            "    history = train_lstm(\n",
            "        model, X_train, X_val,\n",
            "        epochs=50,\n",
            "        batch_size=64,\n",
            "        lr=1e-3,\n",
            "        device=device,\n",
            "    )\n",
            "    torch.save(model.state_dict(), 'lstm_checkpoint.pt')\n",
            "    with open('training_history.json', 'w') as f:\n",
            "        json.dump(history, f, indent=2)\n",
            "    print(\"\\n✓ Model trained and saved.\")"
        ]
    },
    # Training curves
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_training_curves",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Plot Training Curves ───────────────────────────────────────────────────────\n",
            "if history is not None:\n",
            "    from models import plot_training_curves\n",
            "    plot_training_curves(history)\n",
            "    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')\n",
            "else:\n",
            "    print(\"No training history available (loaded from checkpoint).\")"
        ]
    },
    # 2.3 Perplexity header
    {
        "cell_type": "markdown",
        "id": "sec2_eval_header",
        "metadata": {},
        "source": [
            "### 2.3 Perplexity Comparison: Markov vs LSTM\n"
        ]
    },
    # 2.3 Perplexity code
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_eval_ppl",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Test Perplexity ────────────────────────────────────────────────────────────\n",
            "model.eval()\n",
            "criterion = nn.CrossEntropyLoss(reduction='mean')\n",
            "\n",
            "inp_test = torch.from_numpy(X_test[:, :-1, :]).long().to(device)\n",
            "tgt_test = torch.from_numpy(X_test[:, 1:, :]).long().to(device)\n",
            "\n",
            "with torch.no_grad():\n",
            "    logits = model(inp_test)\n",
            "    loss = 0.0\n",
            "    for v in range(4):\n",
            "        loss += criterion(\n",
            "            logits[:, :, v, :].reshape(-1, VOCAB_SIZE),\n",
            "            tgt_test[:, :, v].reshape(-1),\n",
            "        ).item()\n",
            "    lstm_test_ppl = float(np.exp(loss / 4.0))\n",
            "\n",
            "print(f\\\"{'Model':<25} {'Test Perplexity':>15}\\\")\n",
            "print(\\\"-\\\" * 42)\n",
            "print(f\\\"{'Bigram Markov Chain':<25} {markov_test_ppl:>15.2f}\\\")\n",
            "print(f\\\"{'LSTM (2-layer, h=256)':<25} {lstm_test_ppl:>15.2f}\\\")\n",
            "print(f\\\"{'Improvement':<25} {markov_test_ppl / lstm_test_ppl:>15.1f}x\\\")"
        ]
    },
    # 2.4 Generation header
    {
        "cell_type": "markdown",
        "id": "sec2_gen_header",
        "metadata": {},
        "source": [
            "### 2.4 Generation & Comparison\n",
            "\n",
            "We generate chorales from both models and compare:\n",
            "- Token distribution (should approximate real Bach)\n",
            "- Piano roll visualisation\n",
            "- MIDI audio quality\n"
        ]
    },
    # 2.4 Generation code
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_generate",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Generate from LSTM ─────────────────────────────────────────────────────────\n",
            "# Generate the main deliverable at temperature 0.9\n",
            "seed = X_train[10, 0:1, :]\n",
            "lstm_gen = generate_lstm(\n",
            "    model, tokenizer.decode, length=192, temperature=0.9,\n",
            "    seed=seed, device=device\n",
            ")\n",
            "roll_to_midi(lstm_gen, 'symbolic_unconditioned.mid')\n",
            "print(\"★ symbolic_unconditioned.mid exported (192 steps = ~48 beats)\")\n",
            "\n",
            "# Also generate at different temperatures for comparison\n",
            "for temp in [0.7, 1.0, 1.2]:\n",
            "    gen = generate_lstm(\n",
            "        model, tokenizer.decode, length=128, temperature=temp,\n",
            "        seed=X_train[42, 0:1, :], device=device\n",
            "    )\n",
            "    roll_to_midi(gen, f'lstm_chorale_t{temp:.1f}.mid')\n",
            "    print(f\"  ✓ lstm_chorale_t{temp:.1f}.mid exported\")"
        ]
    },
    # 2.4 Distribution comparison
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec2_comparison",
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Distribution Comparison ────────────────────────────────────────────────────\n",
            "# Generate 20 samples from each model for statistical comparison\n",
            "markov_samples = []\n",
            "lstm_samples = []\n",
            "for i in range(20):\n",
            "    seed = X_train[i, 0:1, :]\n",
            "    markov_samples.append(mc.generate(64, temperature=0.8, seed_token=seed[0]))\n",
            "    lstm_samples.append(\n",
            "        generate_lstm(model, None, 64, temperature=0.9, seed=seed, device=device)\n",
            "    )\n",
            "\n",
            "markov_all = np.stack(markov_samples)\n",
            "lstm_all = np.stack(lstm_samples)\n",
            "\n",
            "# Plot token distribution comparison\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)\n",
            "bins = np.arange(VOCAB_SIZE + 1) - 0.5\n",
            "\n",
            "for ax, data, title, color in zip(\n",
            "    axes,\n",
            "    [X_test, markov_all, lstm_all],\n",
            "    ['Real Bach (test)', 'Bigram Markov', 'LSTM (ours)'],\n",
            "    ['#2c3e50', '#e74c3c', '#2ecc71'],\n",
            "):\n",
            "    ax.hist(data.flatten(), bins=bins, density=True, alpha=0.75,\n",
            "            edgecolor='white', color=color, linewidth=0.5)\n",
            "    ax.set_xlabel('Token index')\n",
            "    ax.set_ylabel('Density')\n",
            "    ax.set_title(title, fontweight='bold')\n",
            "    ax.set_xlim(-0.5, VOCAB_SIZE - 0.5)\n",
            "    ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.suptitle('Token Distribution: Real vs Generated', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "# Piano roll comparison\n",
            "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n",
            "titles = ['Real Bach', 'Bigram Markov', 'LSTM']\n",
            "rolls = [\n",
            "    tokenizer.decode(X_test[0]),\n",
            "    tokenizer.decode(markov_all[0]),\n",
            "    tokenizer.decode(lstm_all[0]),\n",
            "]\n",
            "\n",
            "for ax, roll, title in zip(axes, rolls, titles):\n",
            "    for vi, (vname, color) in enumerate(zip(VOICE_NAMES, VOICE_COLORS)):\n",
            "        voice = roll[:, vi]\n",
            "        t = 0\n",
            "        while t < len(voice):\n",
            "            p = voice[t]\n",
            "            if p == 0:\n",
            "                t += 1; continue\n",
            "            dur = 1\n",
            "            while t + dur < len(voice) and voice[t + dur] == p:\n",
            "                dur += 1\n",
            "            ax.barh(p, dur, left=t, height=0.7, color=color, alpha=0.7)\n",
            "            t += dur\n",
            "    ax.set_ylabel('MIDI Pitch')\n",
            "    ax.set_title(title, fontweight='bold')\n",
            "    ax.set_ylim(35, 82)\n",
            "\n",
            "axes[-1].set_xlabel('Time (16th notes)')\n",
            "fig.suptitle('Piano Roll Comparison: Real vs Generated', fontsize=13, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig('piano_roll_comparison.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n✓ All comparisons generated.\")"
        ]
    },
]

# Replace the section 2 placeholder with new cells
nb['cells'] = nb['cells'][:sec2_idx] + new_cells + nb['cells'][sec3_idx:]

# Also update Related Work section to reflect Task 4 instead of Task 2
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'Section 5: Related Work' in src:
        nb['cells'][i]['source'] = [
            "---\n",
            "## Section 5: Related Work\n",
            "\n",
            "### Task 1 — Symbolic Unconditioned Generation\n",
            "\n",
            "| Work | Model | Dataset | Key Contribution |\n",
            "|---|---|---|---|\n",
            "| Allan & Williams (2005) | HMM | Various | Classic baseline with BIC model selection |\n",
            "| DeepBach (Hadjeres 2017) | Gibbs + LSTM | JSB Chorales | Steerable: any voice can be fixed |\n",
            "| BachBot (Liang 2017) | LSTM seq2seq | JSB Chorales | Turing test fooled 1-in-3 listeners |\n",
            "| Music Transformer (Huang 2019) | Transformer + relative attention | Piano-only | Best perplexity via relative positional encoding |\n",
            "| BacHMMachine (Hahn 2021) | Theory-guided HMM | JSB Chorales | Interpretable chord transitions |\n",
            "\n",
            "### Task 4 — Continuous Conditioned Generation\n",
            "\n",
            "| Work | Model | Key Contribution |\n",
            "|---|---|---|\n",
            "| MusicGen (Copet 2023) | Single-stage AR LM over EnCodec | Efficient single-codebook interleaving |\n",
            "| MusicLM (Agostinelli 2023) | Hierarchical audio LM | Semantic + acoustic token hierarchy |\n",
            "| AudioCraft (Meta 2023) | Open-source framework | pip install audiocraft |\n",
            "| FMA Dataset (Defferrard 2017) | - | 106K tracks, 161 genres, CC licensed |\n",
            "| Genre Fine-tuning (IJCRT 2025) | MusicGen-small fine-tuned | Same FMA genre approach as ours |\n",
            "\n",
            "Our work differs in combining both symbolic and continuous tasks,\n",
            "providing a direct comparison of generation paradigms.\n"
        ]
        break

# Write updated notebook
with open('workbook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"✓ workbook.ipynb updated:")
print(f"  - Header updated: Task 1 + Task 4")
print(f"  - Section 2: {len(new_cells)} cells added (Markov + LSTM + training + generation)")
print(f"  - Related Work: updated for Task 4")
print(f"  - Total cells: {len(nb['cells'])}")

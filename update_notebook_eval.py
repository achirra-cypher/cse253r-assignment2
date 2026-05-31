#!/usr/bin/env python3
"""Inject Section 4 (Evaluation) cells into workbook.ipynb."""

import json

with open("workbook.ipynb", "r") as f:
    nb = json.load(f)

# Find Section 4 placeholder and Section 5
sec4_idx = sec5_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "## Section 4: Evaluation" in src:
        sec4_idx = i
    if "## Section 5: Related Work" in src and sec4_idx is not None:
        sec5_idx = i
        break

assert sec4_idx is not None, "Section 4 not found"
assert sec5_idx is not None, "Section 5 not found"
print(f"Replacing Section 4 placeholder (cells {sec4_idx}..{sec5_idx - 1})")

eval_cells = [
    {
        "cell_type": "markdown",
        "id": "sec4_header",
        "metadata": {},
        "source": [
            "---\n",
            "## Section 4: Evaluation\n",
            "\n",
            "We evaluate both tasks with quantitative metrics and qualitative listening.\n",
            "\n",
            "**Task 1:** Perplexity, pitch-class KL divergence, interval histograms,\n",
            "voice range violations, and parallel 5ths/octaves.\n",
            "\n",
            "**Task 4:** Genre classifier consistency (pretrained vs fine-tuned MusicGen)\n",
            "and side-by-side listening comparison.\n",
            "\n",
            "Standalone scripts: `evaluate_task1.py`, `evaluate_task4.py`\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "sec4_task1_header",
        "metadata": {},
        "source": ["### 4.1 Task 1 — Symbolic Evaluation\n"],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec4_task1_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run Task 1 evaluation (or load cached results)\n",
            "import os, json\n",
            "import subprocess\n",
            "\n",
            "if not os.path.exists('evaluation_task1.json'):\n",
            "    subprocess.run(['python', 'evaluate_task1.py', '--n-samples', '30'], check=True)\n",
            "\n",
            "with open('evaluation_task1.json') as f:\n",
            "    eval1 = json.load(f)\n",
            "\n",
            "print('Task 1 Evaluation Results')\n",
            "print('=' * 50)\n",
            "print(f\"{'Metric':<30} {'Markov':>10} {'LSTM':>10}\")\n",
            "print('-' * 50)\n",
            "ppl = eval1['perplexity']\n",
            "print(f\"{'Test perplexity':<30} {ppl['markov']:>10.3f} {ppl['lstm']:>10.3f}\")\n",
            "kl = eval1['pitch_class_kl_to_real']\n",
            "print(f\"{'Pitch-class KL (→ Real)':<30} {kl['markov']:>10.4f} {kl['lstm']:>10.4f}\")\n",
            "l1 = eval1['interval_l1_to_real']\n",
            "print(f\"{'Interval L1 (→ Real)':<30} {l1['markov']:>10.4f} {l1['lstm']:>10.4f}\")\n",
            "vr = eval1['voice_range_violation_rate']\n",
            "print(f\"{'Voice range violations':<30} {vr['markov']:>10.4f} {vr['lstm']:>10.4f}\")\n",
            "pa = eval1['parallel_fifths_octaves_rate']\n",
            "print(f\"{'Parallel 5ths/octaves':<30} {pa['markov']:>10.4f} {pa['lstm']:>10.4f}\")\n",
            "print(f\"\\nReal Bach reference — range: {vr['real']:.4f}, parallel: {pa['real']:.4f}\")\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec4_task1_plots",
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "import os\n",
            "\n",
            "for img in ['eval_pitch_kl.png', 'eval_intervals.png', 'eval_summary.png']:\n",
            "    if os.path.exists(img):\n",
            "        print(img)\n",
            "        display(Image(filename=img))\n",
            "    else:\n",
            "        print(f'  (run evaluate_task1.py to generate {img})')\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "sec4_task1_discussion",
        "metadata": {},
        "source": [
            "**Task 1 Discussion:** The LSTM achieves lower perplexity than the bigram Markov\n",
            "baseline, indicating better next-token prediction. Pitch-class KL and interval\n",
            "distributions measure how Bach-like the generated chorales sound. Voice-leading\n",
            "metrics check whether generated music respects SATB conventions.\n",
            "\n",
            "Qualitative listening: compare `markov_chorale.mid`, `symbolic_unconditioned.mid`,\n",
            "and `sample_chorale_real.mid`.\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "sec4_task4_header",
        "metadata": {},
        "source": ["### 4.2 Task 4 — Continuous Conditioned Evaluation\n"],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec4_task4_code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Task 4 evaluation — genre consistency\n",
            "import os, json\n",
            "import subprocess\n",
            "\n",
            "if not os.path.exists('evaluation_task4.json'):\n",
            "    cmd = ['python', 'evaluate_task4.py']\n",
            "    if not os.path.exists('musicgen_data'):\n",
            "        cmd.append('--skip-classifier')\n",
            "    subprocess.run(cmd, check=False)\n",
            "\n",
            "if os.path.exists('evaluation_task4.json'):\n",
            "    with open('evaluation_task4.json') as f:\n",
            "        eval4 = json.load(f)\n",
            "\n",
            "    print('Task 4 Evaluation — Genre Consistency')\n",
            "    print('=' * 50)\n",
            "    if 'pretrained' in eval4:\n",
            "        print(f\"  Pretrained MusicGen accuracy: {eval4['pretrained']['genre_accuracy']:.3f}\")\n",
            "    if 'finetuned' in eval4:\n",
            "        print(f\"  Fine-tuned MusicGen accuracy:  {eval4['finetuned']['genre_accuracy']:.3f}\")\n",
            "    if 'classifier_error' in eval4:\n",
            "        print(f\"  Note: {eval4['classifier_error']}\")\n",
            "\n",
            "    print('\\nQualitative comparison (pretrained vs fine-tuned):')\n",
            "    for row in eval4.get('comparison', []):\n",
            "        print(f\"  {row['genre']}: {row['prompt']}\")\n",
            "        print(f\"    Pretrained: {row.get('pretrained_file', 'N/A')}\")\n",
            "        print(f\"    Fine-tuned: {row.get('finetuned_file', 'N/A')}\")\n",
            "else:\n",
            "    print('Run prepare_fma_data.py → musicgen_generate.py → evaluate_task4.py')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "sec4_task4_plots",
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display, Audio\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "if os.path.exists('eval_task4_genre_accuracy.png'):\n",
            "    display(Image(filename='eval_task4_genre_accuracy.png'))\n",
            "\n",
            "# Play generated samples if available\n",
            "for d in ['generated_audio', 'generated_audio_finetuned']:\n",
            "    p = Path(d)\n",
            "    if p.exists():\n",
            "        for f in sorted(p.glob('*.mp3'))[:2]:\n",
            "            print(f'Listening: {f}')\n",
            "            display(Audio(filename=str(f)))\n",
            "        for f in sorted(p.glob('*.wav'))[:2]:\n",
            "            print(f'Listening: {f}')\n",
            "            display(Audio(filename=str(f)))\n",
            "\n",
            "if os.path.exists('continuous_conditioned.mp3'):\n",
            "    print('\\n★ Main deliverable: continuous_conditioned.mp3')\n",
            "    display(Audio(filename='continuous_conditioned.mp3'))\n",
        ],
    },
    {
        "cell_type": "markdown",
        "id": "sec4_task4_discussion",
        "metadata": {},
        "source": [
            "**Task 4 Discussion:** We fine-tune MusicGen-small on FMA genre subsets so that\n",
            "text prompts like \"hip hop music with beats\" produce genre-consistent audio.\n",
            "The genre classifier (trained on MFCC features) measures whether generated audio\n",
            "matches the target genre better than the pretrained baseline.\n",
            "\n",
            "Fine-tuning counts as training our own weights — we update the transformer decoder\n",
            "on (audio, text) pairs from FMA-small.\n",
        ],
    },
]

# Replace placeholder Section 4 cell(s) before Section 5
nb["cells"] = nb["cells"][:sec4_idx] + eval_cells + nb["cells"][sec5_idx:]

with open("workbook.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Added {len(eval_cells)} Section 4 cells. Total cells: {len(nb['cells'])}")

#!/usr/bin/env python3
"""Rebuild colab_task4_musicgen.ipynb as a slim Colab reference stub."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

cells.append(new_markdown_cell("""\
# CSE 253R Assignment 2 — Task 4: MusicGen Fine-tuning (Colab Reference)

> **NOTE:** This is a Colab-only training reference.
> The **submission deliverable is `workbook.ipynb`**, which contains all Task 1 and
> Task 4 content in a single unified notebook with all inference, EDA, and evaluation cells.

## Purpose
Fine-tunes MusicGen-small on FMA-small genre data to produce the checkpoint
stored at `task4_weights/`. You do not need to run this for evaluation.

## Checkpoint Mapping
| File | Content |
|---|---|
| `task4_weights/model-008.safetensors` | Best weights (step 144, epoch 4/5) |
| `task4_weights/best/model.safetensors` | Symlink to model-008 (use for inference) |
| `finetune_history.json` | Training/eval loss per step |

## Loading in workbook.ipynb
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
processor = AutoProcessor.from_pretrained("task4_weights/best")
model     = MusicgenForConditionalGeneration.from_pretrained("task4_weights/best")
```
"""))

cells.append(new_code_cell("""\
# Configuration
SMOKE_TEST = True   # True = ~30 min | False = full ~5 hr run
REPO_URL   = "https://github.com/achirra-cypher/cse253r-assignment2.git"
BRANCH     = "main"
USE_DRIVE  = False
"""))

cells.append(new_code_cell("""\
# Step 0 — Check GPU
import subprocess
subprocess.run(["nvidia-smi"])
import torch
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

cells.append(new_code_cell("""\
# Step 1 — Clone repo & install deps
import os
from pathlib import Path
PROJECT_DIR = "/content/cse253r-assignment2"
if Path(PROJECT_DIR).exists():
    os.system(f"rm -rf {PROJECT_DIR}")
os.system(f"git clone -b {BRANCH} {REPO_URL} {PROJECT_DIR}")
os.chdir(PROJECT_DIR)
os.system("pip install -q datasets soundfile librosa transformers accelerate torchaudio scipy scikit-learn")
"""))

cells.append(new_code_cell("""\
# Step 2 — Prepare FMA data
import os
N_SAMPLES = 20 if SMOKE_TEST else 200
os.system(f"python prepare_fma_data.py --n-samples {N_SAMPLES}")
"""))

cells.append(new_code_cell("""\
# Step 3 — Fine-tune MusicGen
import os
EPOCHS = 5 if SMOKE_TEST else 25
os.system(f"python musicgen_finetune.py --epochs {EPOCHS} --batch-size 2")
"""))

cells.append(new_code_cell("""\
# Step 4 — Generate audio
import os
os.system("python musicgen_generate.py --checkpoint finetuned_musicgen --all-genres --out-dir generated_audio_finetuned")
os.system("python musicgen_generate.py --all-genres --out-dir generated_audio")
os.system('python musicgen_generate.py --checkpoint finetuned_musicgen --prompt "hip hop music with beats and rhythm" --output continuous_conditioned.mp3 --duration 30')
"""))

cells.append(new_code_cell("""\
# Step 5 — Listen to results
from IPython.display import Audio, display
from pathlib import Path
for f in sorted(Path("generated_audio_finetuned").glob("*.mp3")):
    print(f.name)
    display(Audio(str(f)))
if Path("continuous_conditioned.mp3").exists():
    print("Main deliverable:")
    display(Audio("continuous_conditioned.mp3"))
"""))

cells.append(new_code_cell("""\
# Step 6 — Download outputs (zip for local use)
import shutil, os
from pathlib import Path
from google.colab import files

staging = Path("/content/task4_outputs")
staging.mkdir(exist_ok=True)

for name in ["continuous_conditioned.mp3", "finetune_history.json",
             "evaluation_task4.json", "eval_task4_genre_accuracy.png"]:
    if Path(name).exists():
        shutil.copy(name, staging / name)

for folder in ["generated_audio", "generated_audio_finetuned"]:
    if Path(folder).exists():
        shutil.copytree(folder, staging / folder, dirs_exist_ok=True)

# NOTE: Rename finetuned_musicgen/ -> task4_weights/ after download to your machine
print("Rename finetuned_musicgen/ -> task4_weights/ in your project after download.")
shutil.make_archive("/content/task4_outputs", "zip", staging)
files.download("/content/task4_outputs.zip")
"""))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

with open("colab_task4_musicgen.ipynb", "w") as f:
    nbformat.write(nb, f)

print("colab_task4_musicgen.ipynb rebuilt as slim reference stub.")
print(f"  Cells: {len(cells)}")

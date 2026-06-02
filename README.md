# CSE 253R Assignment 2 — Music Generation
## Handoff Context File (pass this to any new session for full context)

---

## Assignment Overview

**Goal:** Generate interesting music using ML. Two tasks chosen from four options:

1. Symbolic unconditioned (learn p(music), freely generate MIDI)
2. Symbolic conditioned (generate MIDI given some input)
3. Continuous unconditioned (generate raw audio from scratch)
4. Continuous conditioned (generate audio given text/melody/style input)

**Chosen Tasks:** Task 1 (Symbolic Unconditioned) + Task 4 (Continuous Conditioned)

**Deadline:** ~6 days from when work started (all training complete)

**Deliverables:**
- `workbook.html` — Jupyter notebook exported as HTML (clean, documented)
- `video_url.txt` — URL to ~20 min MP4 presentation (Google Drive or YouTube)
- `symbolic_unconditioned.mid` — generated MIDI from Task 1
- `continuous_conditioned.mp3` — generated audio from Task 4
- Peer grading report (due ~1 week after submission)

**Grading:** 8 marks per task x 2 tasks = 16 marks + 4 marks peer grading = 20 total.
Each task graded on: Data/EDA (2pts), Modeling (2pts), Evaluation (2pts), Related Work (2pts).

---

## Notebook Deliverable

**Single unified notebook:** `workbook.ipynb` (also exported as `workbook.html`)

The notebook covers both tasks with 4 grading sections each:

| Section | Task 1 | Task 4 |
|---------|--------|--------|
| 1. Related Work | T1.1 | T4.1 |
| 2. Data + EDA | T1.2 | T4.2 |
| 3. Modeling | T1.3 | T4.3 |
| 4. Evaluation | T1.4 | T4.4 |

**The colab notebook (`colab_task4_musicgen.ipynb`) is for training only.** It is
not the submission notebook. All Task 4 content has been merged into `workbook.ipynb`.

---

## Quick Start (No Training Needed)

All weights are pre-trained. Just load checkpoints and run inference.

### 1. Install dependencies

```bash
cd cse253r-assignment2
python3 -m venv .venv && source .venv/bin/activate
pip install numpy torch matplotlib music21 pretty_midi scipy \
            soundfile librosa scikit-learn torchaudio transformers accelerate
```

### 2. Regenerate data (fast, no training)

```bash
python3 generate_data.py
```

### 3. Run the notebook

```bash
jupyter notebook workbook.ipynb
# Run all cells — training cells are skipped automatically (checkpoints exist)
```

### 4. Generate Task 1 MIDI from the saved LSTM checkpoint

```bash
python3 -c "
import torch, numpy as np, json
from models import ChoraleLSTM, generate_lstm
from workbook import roll_to_midi  # or copy roll_to_midi from workbook

tokenizer_data = json.load(open('vocab.json'))
# see models.py for details
"
# Or simply re-run cells 20-22 in workbook.ipynb
```

### 5. Generate Task 4 audio from the saved MusicGen checkpoint

```bash
python3 musicgen_generate.py \
    --checkpoint task4_weights/best \
    --prompt "hip hop music with beats and rhythm" \
    --output test.mp3 \
    --duration 15

python3 musicgen_generate.py --checkpoint task4_weights/best --all-genres
```

---

## Weight Paths (Corrected)

### Task 1 — Symbolic LSTM

| File | Description | Location |
|------|-------------|----------|
| `markov_checkpoint.npz` | Bigram Markov chain transition matrices | Repo root (git-tracked) |
| `lstm_checkpoint.pt` | Best LSTM weights (epoch 18, val ppl 1.96) | Repo root (gitignored, ~4.5 MB) |
| `training_history.json` | Loss/perplexity per epoch (23 epochs) | Repo root (git-tracked) |

Load in notebook:
```python
mc = BigramMarkovChain(vocab_size=47)
mc.fit(X_train)   # fast, ~2s on CPU

model = ChoraleLSTM(vocab_size=47, embed_dim=64, hidden_dim=256, n_layers=2, dropout=0.3)
model.load_state_dict(torch.load('lstm_checkpoint.pt', map_location='cpu'))
```

### Task 4 — Fine-tuned MusicGen

| Path | Description | Size |
|------|-------------|------|
| `task4_weights/best/` | Best checkpoint (step 144, epoch 5) | ~2.2 GB |
| `task4_weights/best/model.safetensors` | Model weights (symlink to model-008.safetensors) | 2.2 GB |
| `task4_weights/best/config.json` | Model config | 6 KB |
| `task4_weights/best/generation_config.json` | Generation config | 218 B |
| `task4_weights/best/tokenizer.json` | Tokenizer | 2 MB |
| `task4_weights/checkpoint-36/` through `checkpoint-180/` | Intermediate checkpoints | ~5 GB total |
| `finetune_history.json` | Training/eval loss per epoch | Repo root |

**The checkpoint directory was renamed from `finetuned_musicgen/` to `task4_weights/`.**
Any scripts or documentation using `finetuned_musicgen` should be updated to `task4_weights`.

Load in Python:
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("task4_weights/best")
model = MusicgenForConditionalGeneration.from_pretrained(
    "task4_weights/best",
    torch_dtype=torch.float32
)
model.eval()
```

Do NOT use `task1_weights_download` (that directory does not exist). The correct path is always `task4_weights/`.

---

## Chosen Tasks

### Task 1: Symbolic Unconditioned Generation

**Dataset:** JSB Chorales (Bach 4-part SATB chorales)
**Models:** Bigram Markov chain (baseline) + 2-layer LSTM
**Output:** `symbolic_unconditioned.mid`

Results:

| Model | Test Perplexity |
|-------|----------------|
| Bigram Markov Chain | 2.59 |
| LSTM (2-layer, h=256, 1.1M params) | 1.97 |
| Improvement | 1.3x |

### Task 4: Continuous Conditioned Generation (MusicGen Fine-tuning)

**Dataset:** FMA-small (8,000 tracks x 30s, 8 genres, Creative Commons)
**Model:** Fine-tune MusicGen-small (Meta/AudioCraft, 300M params) on genre subsets
**Output:** `continuous_conditioned.mp3`

Results:

| Model | Genre Classifier Accuracy |
|-------|--------------------------|
| Pretrained MusicGen (no fine-tune) | 25% (1/4 genres) |
| Fine-tuned MusicGen (ours) | 75% (3/4 genres) |

Per-genre (fine-tuned): Folk correct, Electronic correct, Rock correct, Hip-Hop predicted as Electronic.

---

## 6-Day Plan

### Day 1 (DONE) — Data, EDA, Shared Infrastructure
- Loaded 368 Bach chorales via music21 (no download needed)
- Built full preprocessing: tokenizer, piano-roll format, train/val/test split
- Full EDA: voice ranges, pitch class distributions, interval statistics, lengths
- FMA-small EDA: genre distribution, architecture diagram, preprocessing pipeline
- Notebook structure created with all sections

### Day 2 (DONE) — Task 1: Models
- Implemented Bigram Markov Chain baseline (per-voice 47x47 transition matrices, Laplace smoothing)
- Implemented LSTM model: 4xEmbedding(47,64) -> LSTM(256, 2 layers, dropout=0.3) -> 4xLinear(256,47)
- Trained LSTM: 23 epochs (early stopped), Adam lr=1e-3, batch_size=64
- Generated symbolic_unconditioned.mid (192 steps, temperature=0.9)

### Day 3 (DONE) — Task 4: MusicGen Fine-tuning

**Status: Colab smoke test COMPLETE**

- `colab_task4_musicgen.ipynb` — step-by-step Colab notebook (T4 GPU)
- `prepare_fma_data.py` — downloads FMA from HuggingFace, builds WAV + JSON pairs
- `musicgen_finetune.py` — fine-tunes `facebook/musicgen-small` (HuggingFace Trainer)
- `musicgen_generate.py` — generates audio from pretrained or fine-tuned checkpoint
- Fine-tuned 5 epochs on 72 training pairs (20 tracks/genre, 4 genres)
- Generated `continuous_conditioned.mp3` (30s hip-hop prompt)
- Checkpoint saved to `task4_weights/` (renamed from `finetuned_musicgen/`)

### Day 4 (DONE) — Evaluation for Both Tasks

Task 1: `evaluate_task1.py` (30 samples each)
Task 4: `evaluate_task4.py` (genre classifier)

Files: `evaluation_task1.json`, `evaluation_task4.json`, all eval plots

### Day 5 (DONE) — Related Work + Notebook Polish

- All 4 grading sections present for both tasks in `workbook.ipynb`
- Training cells marked "COLAB ONLY / Do not run locally"
- Waveform/spectrogram EDA shows pretrained vs fine-tuned comparison
- Task 4 training curves from `finetune_history.json`
- Related work sections have prose paragraphs + comparison tables
- All 51 cells run without errors (verified with `jupyter nbconvert --execute`)
- Live inference confirmed: fine-tuned MusicGen generates genre-specific WAV files

### Day 6 — Presentation Video + Submission (REMAINING)

Video structure (~20 min):
- 0:00-2:00  — Intro: two tasks, why chosen, dataset overview
- 2:00-10:00 — Task 1: EDA -> Model -> Evaluation -> Related Work
- 10:00-18:00 — Task 4: EDA -> Model -> Evaluation -> Related Work
- 18:00-20:00 — Play generated music from both tasks

Submission checklist:
- workbook.html starts with `<!DOCTYPE html>`
- video_url.txt contains Google Drive or YouTube link
- Video is MP4, >1MB, publicly accessible
- symbolic_unconditioned.mid present
- continuous_conditioned.mp3 present
- Test video: `wget -O test.mp4 '<your_url>'`

---

## Team Members

| Role | Name |
|------|------|
| Speaker A — Task 1 EDA & Preprocessing | **Akhil Teja Chirra** |
| Speaker B — Task 1 Modeling & Evaluation | **Narain Shriraam MS** |
| Speaker C — Task 4 EDA, Preprocessing & Modeling | **Priyansh Parikh** |
| Speaker D — Task 4 Evaluation, Related Work & Conclusion | **Jeevan N V** |

---

## Presentation Materials

- **PRESENTATION.md** — Full slide-by-slide script and talking points for the ~20-min video
- **slides/presentation.pptx** — PowerPoint deck (20 slides, 16:9 widescreen, all images embedded)
- **slides/index.html** — HTML version of the presentation (browser-viewable)

---

## Files Reference

### Day 1-2 (Task 1)

| File | Description | Size |
|------|-------------|------|
| `jsb_chorales.npy` | 368 chorales as (T,4) MIDI pitch arrays | 1.3 MB |
| `jsb_paths.npy` | Corresponding music21 corpus paths | 110 KB |
| `X_train.npy` | 2845 training windows of shape (64,4) | 2.8 MB |
| `X_val.npy` | 178 validation windows of shape (64,4) | 179 KB |
| `X_test.npy` | 190 test windows of shape (64,4) | 191 KB |
| `vocab.json` | 47-token vocab: {midi_pitch: token_index} | 1.4 KB |
| `split.json` | Train/val/test chorale indices (296/36/36) | 1.8 KB |
| `lstm_checkpoint.pt` | Best LSTM model weights (epoch 18) | 4.5 MB |
| `training_history.json` | Loss/perplexity per epoch (23 epochs) | 2 KB |
| `training_curves.png` | Training + validation loss plots | 78 KB |
| `symbolic_unconditioned.mid` | Main deliverable: LSTM-generated chorale | 1.4 KB |
| `markov_chorale.mid` | Markov baseline generated chorale | 604 B |
| `sample_chorale_real.mid` | Real Bach chorale for comparison | 1.3 KB |

### Day 3-5 (Task 4)

| File | Status | Description |
|------|--------|-------------|
| `colab_task4_musicgen.ipynb` | in repo | Colab training notebook (T4 GPU required) |
| `prepare_fma_data.py` | in repo | Download FMA, build MusicGen training pairs |
| `musicgen_finetune.py` | in repo | Fine-tune MusicGen-small |
| `musicgen_generate.py` | in repo | Generate audio samples |
| `evaluate_task4.py` | in repo | Genre classifier evaluation |
| `requirements.txt` | in repo | Python dependencies |
| `continuous_conditioned.mp3` | in repo | Main Task 4 deliverable (~30s) |
| `generated_audio_finetuned/` | in repo | Fine-tuned genre samples |
| `generated_audio/` | in repo | Pretrained baseline samples |
| `task4_weights/` | local only | Fine-tuned checkpoint (~2.2 GB, not on GitHub) |
| `finetune_history.json` | in repo | Training/eval loss log (5 epochs) |
| `evaluation_task4.json` | in repo | Genre classifier results |
| `eval_task4_genre_accuracy.png` | in repo | Accuracy bar chart |

---

## Key Dataset Facts

### JSB Chorales (Task 1)
- Source: music21 built-in corpus (corpus.getComposer('bach'))
- Size: 368 four-part chorales
- Representation: (T, 4) arrays, 16th-note grid (4 steps per beat)
- Vocabulary: 47 tokens (46 unique MIDI pitches 36-81 + 0 for rest/pad)
- Pitch ranges: Soprano 57-81, Alto 53-74, Tenor 48-69, Bass 36-64
- Lengths: 96-772 steps (median 192 = 48 beats)
- Split: 296 train / 36 val / 36 test chorales

### FMA-Small (Task 4)
- Source: rpmon/fma-genre-classification on HuggingFace
- Size: 8,000 tracks x 30 seconds, 1,000 per genre
- Genres: Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, Instrumental
- Fine-tuned genres: Hip-Hop, Folk, Electronic, Rock
- Audio: MP3, 32kHz, mono, 30-second clips
- License: Creative Commons

---

## Fine-tuned MusicGen Checkpoint Details

**Not hosted on GitHub.** Weights are ~2.2 GB (exceeds GitHub 100 MB per-file limit).

### Checkpoint location
- Local path: `task4_weights/` in this repo directory
- Best weights: `task4_weights/best/` (symlinked from model-008.safetensors = step 144)
- Google Drive backup: `My Drive/MusicGen_Finetuned_Weights/` (ask maintainer for link)

### Training summary

| Setting | Value |
|---------|-------|
| Base model | `facebook/musicgen-small` |
| Dataset | FMA: 20 tracks/genre, 4 genres (Hip-Hop, Folk, Electronic, Rock) |
| Train / valid | 72 / 8 pairs |
| Epochs | 5 (smoke test) |
| Batch size | 2 |
| Training steps | 180 |
| Best eval loss | 7.209 (step 144, epoch 4) |
| Genre classifier accuracy | Pretrained 25% -> Fine-tuned 75% (3/4 genres) |

### Files needed for inference

Place in `task4_weights/best/`:

| File | Size | Required |
|------|------|----------|
| `model.safetensors` (or symlink) | 2.2 GB | Yes |
| `config.json` | 6 KB | Yes |
| `generation_config.json` | 218 B | Yes |
| `processor_config.json` | 320 B | Yes |
| `tokenizer_config.json` | 2.4 KB | Yes |
| `tokenizer.json` | 2 MB | Yes |

The `checkpoint-*/` subdirectories are intermediate training snapshots and are NOT
needed for inference.

### Generate audio locally

```bash
cd cse253r-assignment2
source .venv/bin/activate

# Single prompt
python3 musicgen_generate.py \
    --checkpoint task4_weights/best \
    --prompt "hip hop music with beats and rhythm" \
    --output test.mp3 \
    --duration 15

# All target genres
python3 musicgen_generate.py --checkpoint task4_weights/best --all-genres
```

---

## Evaluation Results Summary

### Task 1

| Metric | Markov | LSTM |
|--------|--------|------|
| Test perplexity | 2.59 | 1.97 |
| Pitch-class KL (to real) | 0.176 | 0.198 |
| Interval L1 (to real) | 0.121 | 0.127 |
| Voice range violations | 0.001 | 0.000 |
| Parallel 5ths/octaves | 0.115 | 0.181 |

### Task 4

| Model | Genre Accuracy |
|-------|---------------|
| Pretrained MusicGen | 25% (1/4) |
| Fine-tuned MusicGen (ours) | 75% (3/4) |

Per-genre (fine-tuned): Folk correct, Electronic correct, Rock correct,
Hip-Hop misclassified as Electronic (acoustic overlap).

---

## Important Notes

1. Task 2 (Harmonization) rejected: Same JSB dataset, not interesting enough. Swapped for Task 4.

2. Task 3 (Continuous Unconditioned) rejected: Training GAN/VAE from scratch in 6 days produces noise.

3. Fine-tuning MusicGen counts as "training your own weights": We update transformer decoder
   weights on FMA data. This is clearly explained in the notebook and presentation.

4. 368 chorales loaded (not 382): music21 corpus has 433 Bach works but only 368 have
   exactly 4 parts. Standard JSB-16th pkl has 382 but requires internet download.

5. Sequence stride: 16 for train (75% overlap, more data), 32 for val/test.

6. LSTM early stopped at epoch 23 (patience=10, best at epoch 18).
   Perplexity improvement is modest (1.3x) because 77% of transitions are unisons.

7. Checkpoint directory naming: `task4_weights/` is the canonical name. The older name
   `finetuned_musicgen/` from training is now deprecated. `task1_weights_download/`
   does not exist and should never be referenced.

8. For Day 6 video: Record with Zoom self-recording or OBS, export as MP4.
   Google Drive sharing: "Anyone with the link" -> copy link -> paste in video_url.txt.

---

## Next Steps

### NOW — Day 6: Video + Final Submission

| Deliverable | Status |
|-------------|--------|
| `workbook.ipynb` (51 cells, all executed) | Done |
| `workbook.html` (export from notebook) | Export needed |
| `symbolic_unconditioned.mid` | Done |
| `continuous_conditioned.mp3` | Done |
| `video_url.txt` | Record ~20 min video, upload, add link |

```bash
# Export HTML for submission
jupyter nbconvert --to html workbook.ipynb
```

### Optional
- Full Colab run: `SMOKE_TEST = False`, `--n-samples 200`, `--epochs 25` for stronger fine-tune
- Upload checkpoint to HuggingFace Hub for easy sharing
- Re-export `workbook.html` after any notebook edits

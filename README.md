# CSE 253R Assignment 2 — Music Generation
## Context & Progress File (pass this to any new session for full context)

---

## Assignment Overview

**Goal:** Generate interesting music using ML. Choose any **2 of 4 tasks**:
1. Symbolic unconditioned (learn p(music), freely generate MIDI)
2. Symbolic conditioned (generate MIDI given some input)
3. Continuous unconditioned (generate raw audio from scratch)
4. Continuous conditioned (generate audio given text/melody/style input)

**Deadline:** ~6 days from when work started (Day 1 is done)

**Deliverables:**
- `workbook.html` — Jupyter notebook exported as HTML (clean, documented)
- `video_url.txt` — URL to ~20 min MP4 presentation (Google Drive or YouTube)
- `symbolic_unconditioned.mid` — generated MIDI from Task 1
- `continuous_conditioned.mp3` — generated audio from Task 4
- Peer grading report (due ~1 week after submission)

**Grading:** 8 marks per task × 2 tasks = 16 marks + 4 marks peer grading = 20 total.
Each task graded on: Data/EDA (2pts), Modeling (2pts), Evaluation (2pts), Related Work (2pts).

---

## Chosen Tasks (after thorough analysis)

### Task 1: Symbolic Unconditioned Generation
**Dataset:** JSB Chorales (Bach 4-part SATB chorales)
**Model:** Bigram Markov chain (baseline) + LSTM
**Output:** `symbolic_unconditioned.mid`
**Why chosen:**
- Dataset loads instantly from music21 built-in corpus (zero download)
- Tiny vocabulary (47 tokens), fast training
- Excellent evaluation story: perplexity, voice-leading rules, ground truth comparison
- Clear related work: DeepBach, BachBot, Music Transformer
- All infrastructure already built on Day 1

### Task 4: Continuous Conditioned Generation (MusicGen Fine-tuning)
**Dataset:** FMA-small (8,000 tracks x 30s, 8 genres, Creative Commons)
**Model:** Fine-tune MusicGen-small (Meta/AudioCraft, 300M params) on genre subsets
**Output:** `continuous_conditioned.mp3`
**Why chosen:**
- Generates real, full-quality audio — most interesting output of all 4 options
- Text conditioning ("upbeat electronic music") enables diverse, varied generation
- Fine-tuning satisfies "train your own weights" requirement
- FMA-small available on HuggingFace (rpmon/fma-genre-classification)
- Strong contrast with Task 1 for presentation narrative
- Related work: MusicGen (Copet 2023), genre fine-tuning (IJCRT 2025)

**Why NOT Tasks 2 or 3:**
- Task 2 (Harmonization): still JSB chorales, still sounds like Bach — not interesting enough
- Task 3 (Continuous Unconditioned from scratch): GAN/VAE produces noise in 6 days — high failure risk

---

## 6-Day Plan

### Day 1 (DONE) — Data, EDA, Shared Infrastructure
- Loaded 368 Bach chorales via music21 (no download needed)
- Built full preprocessing: tokenizer, piano-roll format, train/val/test split
- Full EDA: voice ranges, pitch class distributions, interval statistics, lengths
- FMA-small EDA: genre distribution, architecture diagram, preprocessing pipeline
- Notebook structure created with all sections placeholder
- Files saved: jsb_chorales.npy, X_train/val/test.npy, vocab.json, split.json

### Day 2 (DONE) — Task 1: Models
- Installed dependencies: music21, pretty_midi
- Regenerated all preprocessed data files (generate_data.py)
- Implemented Bigram Markov Chain baseline (per-voice 47×47 transition matrices, Laplace smoothing)
- Implemented LSTM model: 4×Embedding(47,64) → LSTM(256, 2 layers, dropout=0.3) → 4×Linear(256,47)
- Trained LSTM: 28 epochs (early stopped), Adam lr=1e-3, batch_size=64
- Generated symbolic_unconditioned.mid (192 steps, temperature=0.9)
- Generated comparison MIDI files at temperatures 0.7, 0.9, 1.0, 1.2
- Created distribution comparison plots and training curves
- Updated workbook.ipynb: header fixed (Task 1+4), Section 2 populated (11 cells), Related Work updated

Results:
| Model | Test Perplexity |
|-------|----------------|
| Bigram Markov Chain | 2.59 |
| LSTM (2-layer, h=256, 1.1M params) | 1.99 |
| Improvement | 1.3x |

Output: symbolic_unconditioned.mid (from LSTM, 192 steps = ~48 beats)

### Day 3 — Task 4: MusicGen Fine-tuning Setup

**Status: infrastructure DONE · GPU execution REMAINING**

#### Done (local / in repo)
- `colab_task4_musicgen.ipynb` — step-by-step Colab notebook (T4 GPU)
- `prepare_fma_data.py` — downloads FMA from HuggingFace, builds WAV + JSON pairs
- `musicgen_finetune.py` — fine-tunes `facebook/musicgen-small` (HuggingFace Trainer)
- `musicgen_generate.py` — generates audio from pretrained or fine-tuned checkpoint
- `requirements.txt` — dependency list
- `generate_data.py` fix — music21 10.x compatibility (`part.recurse()` instead of `part.flat`)

#### Remaining (run in Colab — see `colab_task4_musicgen.ipynb`)
1. Open notebook in [Google Colab](https://colab.research.google.com), enable **T4 GPU**
2. Set `SMOKE_TEST = True` first (~30 min), then `False` for full run (~4–6 hrs)
3. Run all cells → download `task4_outputs.zip` to your Mac

```bash
# What Colab runs (for reference):
pip install datasets soundfile librosa transformers accelerate torchaudio scikit-learn
python prepare_fma_data.py --n-samples 200
python musicgen_finetune.py --epochs 25 --batch-size 2
python musicgen_generate.py --checkpoint finetuned_musicgen --all-genres
python musicgen_generate.py --checkpoint finetuned_musicgen --output continuous_conditioned.mp3
```

**Target genres:** Hip-Hop, Folk, Electronic, Rock (200 tracks each = 800 pairs)

**Outputs still needed:** `musicgen_data/`, `finetuned_musicgen/`, `generated_audio/`, `continuous_conditioned.mp3`

### Day 4 — Evaluation for Both Tasks

**Status: Task 1 DONE · Task 4 REMAINING (blocked on Day 3 Colab run)**

#### Done
- `evaluate_task1.py` — full Task 1 metrics script
- `evaluate_task4.py` — genre classifier + pretrained vs fine-tuned comparison
- `update_notebook_eval.py` — Section 4 injected into `workbook.ipynb`
- **Task 1 evaluation run locally** — results below

Task 1 results (`evaluate_task1.py`, 30 samples):
| Metric | Markov | LSTM |
|--------|--------|------|
| Test perplexity | 2.59 | 1.97 |
| Pitch-class KL → Real | 0.176 | 0.198 |
| Interval L1 → Real | 0.121 | 0.127 |
| Voice range violations | 0.001 | 0.000 |
| Parallel 5ths/octaves | 0.115 | 0.181 |

Files produced: `evaluation_task1.json`, `eval_pitch_kl.png`, `eval_intervals.png`, `eval_summary.png`

#### Remaining
**Task 4 evaluation** (after Colab produces audio):
```bash
source .venv/bin/activate
python evaluate_task4.py
jupyter notebook workbook.ipynb   # run Section 4 cells
```
- Genre consistency: MFCC classifier on generated audio
- Comparison: pretrained MusicGen vs fine-tuned
- Side-by-side listening
- Outputs needed: `evaluation_task4.json`, `eval_task4_genre_accuracy.png`

**Notebook:** Run Section 4 cells to display Task 1 + Task 4 results together

### Day 5 — Related Work + Notebook Polish
- Task 1 related work: Markov (Allan & Williams 2005), DeepBach (Hadjeres 2017),
  BachBot (Liang 2017), BacHMMachine (Hahn 2021), Music Transformer (Huang 2019)
- Task 4 related work: MusicGen (Copet 2023), AudioCraft, MusicLM (Agostinelli 2023),
  FMA dataset (Defferrard 2017), genre fine-tuning (IJCRT 2025)
- Clean notebook: markdown explanations, plots, remove debug output
- Export: jupyter nbconvert --to html workbook.ipynb
- Verify HTML opens in browser starts with <!DOCTYPE html>

### Day 6 — Presentation Video + Submission
Video structure (~20 min):
- 0:00-2:00  — Intro: two tasks, why chosen, dataset overview
- 2:00-10:00 — Task 1: EDA -> Model -> Evaluation -> Related Work
- 10:00-18:00 — Task 4: EDA -> Model -> Evaluation -> Related Work
- 18:00-20:00 — Play generated music from both tasks

Submission checklist:
- workbook.html starts with <!DOCTYPE html>
- video_url.txt contains Google Drive or YouTube link
- Video is MP4, >1MB, publicly accessible
- symbolic_unconditioned.mid present
- continuous_conditioned.mp3 present
- Test video: wget -O test.mp4 '<your_url>'

---

## Files Created on Day 1

| File | Description | Size |
|------|-------------|------|
| jsb_chorales.npy | 368 chorales as (T,4) MIDI pitch arrays | 1.3MB |
| jsb_paths.npy | Corresponding music21 corpus paths | 110KB |
| X_train.npy | 2845 training windows of shape (64,4) | 2.8MB |
| X_val.npy | 178 validation windows of shape (64,4) | 179KB |
| X_test.npy | 190 test windows of shape (64,4) | 191KB |
| vocab.json | 47-token vocab: {midi_pitch: token_index} | 1.4KB |
| split.json | Train/val/test chorale indices (296/36/36) | 1.8KB |
| workbook.ipynb | Notebook with EDA + placeholder sections | ~30KB |
| test_chorale.mid | Sample MIDI export to verify pipeline | 1.4KB |

---

## Files Created on Day 2

| File | Description | Size |
|------|-------------|------|
| generate_data.py | Standalone data preprocessing script (reproduces Day 1 outputs) | 5.4KB |
| models.py | BigramMarkovChain + ChoraleLSTM + training + generation code | 18.6KB |
| train_and_generate.py | End-to-end training pipeline script | 7.5KB |
| update_notebook.py | Script to inject Section 2 cells into workbook.ipynb | 10KB |
| lstm_checkpoint.pt | Best LSTM model weights (epoch 18) | 4.5MB |
| training_history.json | Loss/perplexity per epoch (28 epochs) | 2KB |
| training_curves.png | Training & validation loss + perplexity plots | 78KB |
| distribution_comparison.png | Token distribution: Real vs Markov vs LSTM | 49KB |
| symbolic_unconditioned.mid | ★ Main deliverable — LSTM-generated chorale (192 steps) | 1.4KB |
| markov_chorale.mid | Markov baseline generated chorale (128 steps) | 604B |
| lstm_chorale_t0.7.mid | LSTM generation at temperature 0.7 | 681B |
| lstm_chorale_t0.9.mid | LSTM generation at temperature 0.9 | 914B |
| lstm_chorale_t1.0.mid | LSTM generation at temperature 1.0 | 799B |
| lstm_chorale_t1.2.mid | LSTM generation at temperature 1.2 | 1.1KB |
| sample_chorale_real.mid | Real Bach chorale for comparison | 1.3KB |

---

## Files Created on Day 3–4

### Day 3 — scripts in repo (outputs pending Colab run)

| File | Status | Description |
|------|--------|-------------|
| colab_task4_musicgen.ipynb | ✅ in repo | Colab notebook — run this on T4 GPU |
| prepare_fma_data.py | ✅ in repo | Download FMA, build MusicGen training pairs |
| musicgen_finetune.py | ✅ in repo | Fine-tune MusicGen-small |
| musicgen_generate.py | ✅ in repo | Generate audio samples |
| requirements.txt | ✅ in repo | Python dependencies |
| musicgen_data/ | ⏳ pending | FMA audio + JSON pairs (after Colab Step 3) |
| finetuned_musicgen/ | ⏳ pending | Fine-tuned checkpoint (after Colab Step 4) |
| generated_audio/ | ⏳ pending | Pretrained baseline samples (after Colab Step 5) |
| generated_audio_finetuned/ | ⏳ pending | Fine-tuned samples (after Colab Step 5) |
| continuous_conditioned.mp3 | ⏳ pending | ★ Main Task 4 deliverable (after Colab Step 5) |

### Day 4 — scripts + Task 1 results in repo

| File | Status | Description |
|------|--------|-------------|
| evaluate_task1.py | ✅ in repo | Task 1 evaluation script |
| evaluate_task4.py | ✅ in repo | Task 4 evaluation script |
| update_notebook_eval.py | ✅ in repo | Injects Section 4 into workbook.ipynb |
| workbook.ipynb Section 4 | ✅ done | Evaluation cells added |
| evaluation_task1.json | ✅ done | Task 1 metrics (Markov vs LSTM) |
| eval_pitch_kl.png | ✅ done | Pitch-class distribution plot |
| eval_intervals.png | ✅ done | Melodic interval histogram |
| eval_summary.png | ✅ done | Summary bar chart |
| evaluation_task4.json | ⏳ pending | After Colab + evaluate_task4.py |
| eval_task4_genre_accuracy.png | ⏳ pending | After Colab + evaluate_task4.py |

---

## Key Dataset Facts

### JSB Chorales (Task 1)
- Source: music21 built-in corpus (corpus.getComposer('bach'))
- Size: 368 four-part chorales (some of 433 Bach works lack 4 parts)
- Representation: (T, 4) arrays, 16th-note grid (4 steps per beat)
- Vocabulary: 47 tokens — 46 unique MIDI pitches (36-81) + 0 (rest/pad)
- Pitch ranges: Soprano 57-81, Alto 53-74, Tenor 48-69, Bass 36-64
- Lengths: 96-772 steps (median 192 = 48 beats = ~12 bars)
- Key insight: 77% of transitions are unisons (sustained notes); 17% stepwise
- Split: 296 train / 36 val / 36 test chorales
- Sequences: 2845 train / 178 val / 190 test (64 steps x 4 voices each)

### FMA-Small (Task 4)
- Source: rpmon/fma-genre-classification on HuggingFace, or fma_small.zip (~8GB)
- Size: 8,000 tracks x 30 seconds, 1,000 per genre
- Genres: Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, Instrumental
- Audio: MP3, 22050Hz, mono, 30-second clips
- License: Creative Commons (legally usable for research)
- Fine-tune plan: 4 genres x 200 tracks = 800 (audio, text_prompt) pairs

---

## Shared Infrastructure (reusable functions)

### Tokenizer (in workbook.ipynb, Section 1)
```python
class Tokenizer:
    vocab_size = 47   # 0=rest, 1-46 = MIDI pitches 36-81
    def encode(self, roll)   # (T,4) MIDI pitches -> (T,4) token indices
    def decode(self, tokens) # (T,4) token indices -> (T,4) MIDI pitches
```

### Sequence Builder
```python
def make_sequences(chorales, tokenizer, seq_len=64, stride=16, split_idx=None)
# Returns (N, 64, 4) integer token array
# seq_len=64 steps = 16 beats = ~4 bars
```

### MIDI Exporter
```python
def roll_to_midi(roll, out_path, bpm=100)
# (T,4) MIDI pitch array -> 4-track .mid file
# Uses pretty_midi library
```

---

## Evaluation Metrics Plan

### Task 1
| Metric | What it measures | Baseline to beat |
|--------|-----------------|------------------|
| Perplexity | How well model predicts test sequences | Markov chain |
| Pitch class KL-divergence | Does generated music use Bach's note frequencies? | Random sampling |
| Voice range violations | % notes outside historical SATB ranges | Markov chain |
| Interval distribution L1 | Are melodic intervals Bach-like? | Markov chain |
| Parallel 5ths/octaves | Voice-leading rule compliance | Markov chain |

### Task 4
| Metric | What it measures | Baseline to beat |
|--------|-----------------|------------------|
| Genre accuracy | Does pretrained classifier agree with target genre? | Pretrained MusicGen (no fine-tune) |
| FAD | Frechet Audio Distance (if compute allows) | Pretrained MusicGen |
| Qualitative | Human listening preference | Pretrained MusicGen + real FMA audio |

---

## Related Work Quick Reference

### Task 1 (Symbolic Unconditioned)
- Allan & Williams 2005: HMM on JSB chorales, classic baseline, BIC model selection
- DeepBach (Hadjeres 2017, ICML): Gibbs sampling + LSTM, steerable (fix any voice)
- BachBot (Liang 2017): LSTM seq2seq, Turing test fooled 1-in-3 listeners
- BacHMMachine (Hahn 2021): Theory-guided HMM, interpretable chord transitions
- Music Transformer (Huang 2019, ICLR): Relative attention, best perplexity on JSB

### Task 4 (Continuous Conditioned)
- MusicGen (Copet 2023, NeurIPS): Single-stage autoregressive LM over EnCodec tokens
- MusicLM (Agostinelli 2023): Hierarchical audio LM with semantic+acoustic tokens
- AudioCraft (Meta 2023): Open-source framework, install: pip install audiocraft
- FMA dataset (Defferrard 2017): Dataset paper with baseline classification results
- Genre fine-tuning (IJCRT 2025): Fine-tuned MusicGen-small on FMA genres, same approach

---

## Important Notes & Decisions

1. Task 2 (Harmonization) rejected: Same JSB dataset, sounds identical to Task 1,
   not interesting enough for presentation graders. Swapped for Task 4.

2. Task 3 (Continuous Unconditioned) rejected: Training GAN/VAE on audio from scratch
   in 6 days produces noise. High failure risk, poor output quality.

3. Fine-tuning MusicGen counts as "training your own weights": We update transformer
   decoder weights on FMA data. Must explain this clearly in presentation.

4. 368 chorales loaded (not 382): music21 corpus includes 433 Bach works but only
   368 have exactly 4 parts. Standard JSB-16th pkl has 382 but needs internet.

5. Sequence stride: 16 for train (75% overlap, more data), 32 for val/test (less overlap).

6. For Day 3 (MusicGen): Run on Colab T4 GPU. Install: pip install audiocraft.
   AudioCraft training API: audiocraft.train with solver=musicgen/musicgen_base_32khz.

7. For Day 6 video: Record with Zoom self-recording or OBS, export as MP4.
   Google Drive sharing: "Anyone with the link" -> copy link -> paste in video_url.txt.

8. Day 2 LSTM early stopped at epoch 28 (patience=10, best at epoch 18).
   Perplexity improvement over Markov is modest (1.3x) partly because 77% of
   transitions are unisons — even a bigram model captures sustained notes well.

9. Notebook header corrected: was "Task 1 + Task 2", now "Task 1 + Task 4".
   Related Work section also updated to cover Task 4 (MusicGen, AudioCraft, FMA).

---

## Next Steps (Day 3+)

### NOW — Day 3: Run Colab notebook (GPU required)
1. Open `colab_task4_musicgen.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. `SMOKE_TEST = True` first, then `False` for full run
4. Download `task4_outputs.zip` when done

### THEN — Day 4: Finish Task 4 evaluation
```bash
source .venv/bin/activate
python evaluate_task4.py
jupyter notebook workbook.ipynb   # run Section 4 cells
```

### Day 5: Polish
- Clean notebook, add markdown explanations
- Export workbook.html
- Write Related Work prose

### Day 6: Video + Submit
- Record ~20 min presentation
- Upload to Google Drive/YouTube
- Final submission checklist

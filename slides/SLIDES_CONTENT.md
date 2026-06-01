# Slide Deck Content Specification
## CSE 253R Assignment 2 — Music Generation

**Purpose:** This file specifies every slide for the presentation deck. A builder
agent will use this to create HTML slides via the frontend-slides skill.
Do NOT deviate from the content or ordering below.

**Design notes for the builder:**
- Dark theme preferred (deep navy or charcoal background, white/light text)
- Accent colors: red (#e74c3c) for Task 1 elements, blue (#3498db) for Task 4 elements
- Code snippets in monospace with syntax highlighting
- Tables should be clean, minimal borders
- Use subtle slide transitions (fade or none), not distracting animations
- Target aspect ratio: 16:9
- Font: system sans-serif stack, 28-32pt for headings, 18-22pt for body
- Total slides: 20

---

## Slide 1 — Title

**Speaker:** A
**Time:** 0:00

**Title (large, centered):**
Music Generation: Symbolic & Continuous

**Subtitle:**
CSE 253R Assignment 2

**Bottom line:**
Task 1: Symbolic Unconditioned Generation | Task 4: Continuous Conditioned Generation

**Footer:**
[Team member names] | Spring 2026

**Narration script:**
"Hi, we're [team names]. For this assignment we tackled two music generation tasks."

---

## Slide 2 — Two Tasks Overview

**Speaker:** A
**Time:** 0:15

**Title:** What We Built

**Layout:** Two-column comparison

**Left column (red accent):**
- Heading: Task 1: Symbolic Unconditioned
- Icon suggestion: MIDI note / musical staff
- Bullet 1: Learn p(Soprano, Alto, Tenor, Bass) on Bach chorales
- Bullet 2: Generate new 4-part MIDI harmony from scratch
- Bullet 3: Models: Bigram Markov Chain, 2-layer LSTM
- Bullet 4: Dataset: JSB Chorales (368 chorales, music21)
- Bullet 5: Output: symbolic_unconditioned.mid

**Right column (blue accent):**
- Heading: Task 4: Continuous Conditioned
- Icon suggestion: waveform / speaker
- Bullet 1: Fine-tune MusicGen-small on genre-labeled audio
- Bullet 2: Generate 30s audio from text prompts
- Bullet 3: Model: MusicGen (300M params, Meta/AudioCraft)
- Bullet 4: Dataset: FMA-small (8,000 tracks, 8 genres)
- Bullet 5: Output: continuous_conditioned.mp3

**Bottom banner:**
Why this pairing? Symbolic = interpretable, small data, rule-checkable.
Continuous = perceptually realistic, requires large pretrained model.

**Narration script:**
"Task 1 is symbolic unconditioned generation, where we train models on Bach chorales represented as sequences of MIDI note tokens and generate new four-part harmony from scratch. Task 4 is continuous conditioned generation, where we fine-tune Meta's MusicGen model on the Free Music Archive to generate raw audio waveforms from text prompts like 'upbeat electronic music with synths.' We picked this combination intentionally. The symbolic task gives us a small, controlled problem with interpretable evaluation metrics. The continuous task gives us perceptually realistic output but requires working with a 300-million parameter pretrained model. The contrast between these two approaches is the throughline of our presentation."

---

## Slide 3 — JSB Chorales Dataset

**Speaker:** A
**Time:** 1:30

**Title:** Task 1 Dataset: JSB Chorales

**Layout:** Key facts on left, small piano-roll image placeholder on right

**Key facts (left, as a styled table or list):**

| Property | Value |
|----------|-------|
| Source | music21 built-in corpus (no download) |
| Composer | J.S. Bach (BWV 1-438) |
| Chorales loaded | 368 (4-part SATB only) |
| Representation | (T, 4) MIDI pitch matrix, 16th-note grid |
| Vocabulary | 47 tokens (46 pitches + rest) |
| Pitch range | MIDI 36 (C2) to MIDI 81 (A5) |
| Split | 296 train / 36 val / 36 test chorales |
| Sequences | 2,845 train / 178 val / 190 test windows |
| Window size | 64 steps = 16 beats = ~4 bars |

**Right side:** Placeholder for piano-roll screenshot (from notebook Cell 4 output).
Use image `eda_piano_roll.png` if available, otherwise a placeholder box labeled
"Piano Roll Visualization (shown in notebook)".

**Bottom note (smaller text):**
Benchmark used in: DeepBach (Hadjeres 2017), BachBot (Liang 2017),
Music Transformer (Huang 2019), BacHMMachine (Hahn 2021)

**Narration script:**
"Our dataset for Task 1 is the JSB Chorales, 368 four-part harmonizations written by Johann Sebastian Bach. Each chorale is an SATB arrangement of a Lutheran hymn. We load them directly from the music21 library's built-in corpus, so there's no external download step. We represent each chorale as a time-by-four matrix of MIDI pitch values, quantized to a sixteenth-note grid. The vocabulary is small, 47 tokens total. We split the 368 chorales into 296 for training, 36 for validation, and 36 for testing."

---

## Slide 4 — Task 1 Modeling Overview

**Speaker:** B
**Time:** 3:45

**Title:** Task 1: Modeling Approach

**Layout:** Two model boxes side by side

**Left box (labeled "Baseline"):**
- Heading: Bigram Markov Chain
- 4 independent 47x47 transition matrices (one per voice)
- Laplace (add-1) smoothing
- Context: current timestep only
- Formula: P(next | current) = (count + 1) / (sum + V)
- Test perplexity: **2.59**

**Right box (labeled "Our Model"):**
- Heading: 2-Layer LSTM
- Joint processing of all 4 voices
- 4 x Embedding(47, 64) concatenated to 256-dim input
- LSTM(256, 256, 2 layers, dropout=0.3)
- 4 x Linear(256, 47) output heads
- Parameters: **1.1M**
- Test perplexity: **1.97** (1.3x improvement)

**Bottom:**
Objective: Cross-entropy loss summed over 4 voices.
Training: Teacher forcing. Generation: Autoregressive sampling with temperature.

**Narration script:**
"The modeling problem is: given a sequence of four-voice tokens at time t, predict the four tokens at time t+1. We optimize cross-entropy loss summed across the four voices. We implemented two models. First, a Bigram Markov Chain as a baseline that learns four independent 47-by-47 transition matrices with Laplace smoothing. Second, a two-layer LSTM that processes all four voices jointly and can capture longer-range dependencies through its hidden state."

---

## Slide 5 — LSTM Architecture Diagram

**Speaker:** B
**Time:** 5:15

**Title:** LSTM Architecture

**Layout:** Vertical flowchart, centered

**Diagram (top to bottom):**

```
Input: (batch, T, 4) token indices
         |
    +-----------+-----------+-----------+-----------+
    | Embed(47,64) | Embed(47,64) | Embed(47,64) | Embed(47,64) |
    |  Soprano   |    Alto    |   Tenor   |    Bass   |
    +-----------+-----------+-----------+-----------+
         |              |            |            |
         +---------- concatenate ----------------+
                        |
                   (batch, T, 256)
                        |
              LSTM(256 in, 256 hidden)
              2 layers, dropout=0.3
                        |
                   (batch, T, 256)
                        |
    +-----------+-----------+-----------+-----------+
    | Linear(256,47) | Linear(256,47) | Linear(256,47) | Linear(256,47) |
    |  Soprano   |    Alto    |   Tenor   |    Bass   |
    +-----------+-----------+-----------+-----------+
         |              |            |            |
    logits: (batch, T, 4, 47)
```

**Side annotation:**
- Teacher forcing during training
- Autoregressive sampling at generation
- Temperature 0.9 for main deliverable
- Early stopped at epoch 28 (best at epoch 18)

**Narration script:**
"Our LSTM takes the four voice tokens at each timestep through four separate embedding layers, each mapping the 47-token vocabulary into a 64-dimensional space. We concatenate those four embeddings into a 256-dimensional vector, feed it through a two-layer LSTM with hidden size 256 and dropout 0.3, and then project back out through four separate linear heads, one per voice. Total parameter count is about 1.1 million."

---

## Slide 6 — Task 1 Evaluation Metrics

**Speaker:** B
**Time:** 7:00

**Title:** Task 1: Evaluation Framework

**Layout:** Table with metric descriptions

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| Test Perplexity | Statistical fit (next-token prediction quality) | Lower = model predicts Bach better |
| Pitch-Class KL Divergence | Note frequency distribution vs real Bach | Checks if generated music uses Bach's pitch palette |
| Interval L1 Distance | Melodic jump pattern vs real Bach | Checks if stepwise/leapwise motion is realistic |
| Voice Range Violations | % notes outside historical SATB ranges | Basic constraint: Soprano should not go below Alto |
| Parallel 5ths/Octaves | Voice-leading rule violations | Music theory constraint Bach almost never violates |

**Bottom question (highlighted):**
Core question: Does better perplexity = better music?

**Narration script:**
"We evaluate with five metrics that capture different aspects of musical quality. Perplexity measures statistical fit. Pitch-class KL divergence checks whether generated music uses the same distribution of note classes as real Bach. Interval L1 measures whether the pattern of melodic jumps matches. Voice range violations count notes outside the historical SATB ranges. And parallel fifths and octaves count violations of a specific voice-leading rule from music theory. The key question is: does better perplexity translate to better music?"

---

## Slide 7 — Task 1 Results and Discussion

**Speaker:** B
**Time:** 8:15

**Title:** Task 1: Results and Discussion

**Layout:** Results table on top, discussion points below

**Results table (30 generated samples per model):**

| Metric | Markov | LSTM | Real Bach |
|--------|--------|------|-----------|
| Test Perplexity | 2.59 | **1.97** | - |
| Pitch-Class KL (to real) | **0.176** | 0.198 | 0 |
| Interval L1 (to real) | **0.121** | 0.127 | 0 |
| Voice Range Violations | 0.09% | **0.01%** | 0% |
| Parallel 5ths/Octaves | **11.5%** | 18.1% | 3.0% |

**Discussion points (below table):**
- LSTM wins on perplexity and voice range compliance
- Markov wins on distributional similarity and parallel motion
- The LSTM optimizes cross-entropy (statistical fit), not voice-leading rules
- Markov chain avoids parallel motion by mostly sustaining notes (77% unisons)
- Consistent with literature: BachBot (similar LSTM) reports comparable perplexity.
  Music Transformer achieves ~1.2 ppl with relative attention.

**Narration script:**
"The LSTM wins on perplexity, 1.97 versus 2.59. But the Markov chain actually scores better on pitch-class KL divergence and has fewer parallel fifths and octaves. The LSTM optimizes cross-entropy, a statistical objective. It does not have an explicit penalty for voice-leading violations. The Markov chain produces blander output and leans heavily on sustaining notes, which accidentally avoids parallel motion. This tension between statistical fit and musical quality is well documented in the literature."

---

## Slide 8 — Task 4 Title Card

**Speaker:** C
**Time:** 9:00

**Title (large):** Task 4: Continuous Conditioned Generation

**Subtitle:** Fine-tuning MusicGen on FMA Genre Data

**Visual:** Conceptual flow arrow:
Text Prompt --> [MusicGen] --> 30s Audio Clip

**Example prompt shown:**
"upbeat electronic music with synthesizers" --> [waveform graphic]

**Narration script:**
"Task 4 is continuous conditioned generation. The input is a text prompt describing a musical genre and style, and the output is a 30-second audio clip, actual playable music, not MIDI tokens."

---

## Slide 9 — FMA-Small Dataset

**Speaker:** C
**Time:** 9:15

**Title:** Task 4 Dataset: FMA-Small (Free Music Archive)

**Layout:** Stats on left, genre list on right

**Left column (key facts):**

| Property | Value |
|----------|-------|
| Source | HuggingFace (rpmon/fma-genre-classification) |
| Total tracks | 8,000 |
| Per genre | 1,000 (perfectly balanced) |
| Clip duration | 30 seconds |
| Sample rate | 22,050 Hz mono |
| Total audio | ~67 hours |
| License | Creative Commons |
| Reference | Defferrard et al. 2017 |

**Right column (genre list with color dots):**
- Hip-Hop
- Pop
- Folk
- Experimental
- Rock
- International
- Electronic
- Instrumental

**Highlight box:**
Fine-tuning subset: 4 genres (Hip-Hop, Folk, Electronic, Rock)
20 tracks/genre = 72 train + 8 val pairs (smoke test on Colab T4)

**Narration script:**
"Our dataset is FMA-small, the Free Music Archive. It contains 8,000 tracks, each 30 seconds long, evenly distributed across 8 genres. For our smoke test on Colab, we used a subset of 4 genres with 20 tracks per genre, giving us 72 training pairs and 8 validation pairs. This is a small training set, and we will come back to that when we discuss limitations."

---

## Slide 10 — Fine-tuning vs Training from Scratch

**Speaker:** C
**Time:** 10:30

**Title:** Why Fine-tune Instead of Training from Scratch?

**Layout:** Two-column comparison table

| Criterion | Train from Scratch | Fine-tune MusicGen (Ours) |
|-----------|-------------------|--------------------------|
| Audio quality | Noise/blur in 6 days | Studio-quality audio |
| Data needed | 100s of hours | ~67 hrs (FMA-small) |
| GPU time | Weeks | ~4-8 hours (Colab T4) |
| Output variety | Usually one mode | 8 distinct genres |
| Trains own weights | Yes | Yes (decoder fine-tuning) |

**Use red background for "Train from Scratch" column, green for "Fine-tune" column.**

**Bottom callout:**
Fine-tuning satisfies the "train your own weights" requirement.
We update the transformer decoder parameters on our genre-labeled data.

**Narration script:**
"Training a continuous audio generation model from scratch would require hundreds of hours of audio and weeks of GPU time. With 6 days and a free Colab T4, that was not realistic. Fine-tuning MusicGen gives us studio-quality audio out of the box because the pretrained model already knows how to produce coherent waveforms. The key point is that fine-tuning still counts as training our own weights."

---

## Slide 11 — MusicGen Architecture

**Speaker:** C
**Time:** 11:30

**Title:** MusicGen Architecture (Copet et al., NeurIPS 2023)

**Layout:** Horizontal pipeline diagram, left to right

**Pipeline blocks (5 boxes connected by arrows):**

1. **Text Prompt** (blue box)
   - "upbeat electronic music with synths"

2. **T5 Text Encoder** (purple box, labeled "FROZEN")
   - T5-base, 768-dim embeddings

3. **Transformer Decoder LM** (red box, labeled "FINE-TUNED", star icon)
   - 300M parameters
   - 24 layers, 16 attention heads
   - Hidden size 1024, FFN 4096

4. **EnCodec Decoder** (green box, labeled "FROZEN")
   - 32 kHz, 4 codebooks
   - Codebook size 2048

5. **Audio Output** (gold box)
   - 32 kHz waveform, 30s

**Annotation arrow pointing to box 3:**
"Only this component is fine-tuned on FMA data"

**Narration script:**
"MusicGen has three components. A frozen T5 text encoder that processes the input prompt. A transformer decoder with 300 million parameters, the language model that generates audio token sequences. And a frozen EnCodec decoder that converts discrete tokens back into a 32 kHz waveform. Only the transformer decoder gets fine-tuned. The text encoder and audio codec stay frozen."

---

## Slide 12 — Fine-tuning Training Details

**Speaker:** C
**Time:** 13:15

**Title:** MusicGen Fine-tuning: Training Summary

**Layout:** Table on left, loss curve placeholder on right

**Training config table:**

| Setting | Value |
|---------|-------|
| Base model | facebook/musicgen-small |
| Training data | 72 pairs (4 genres x 18 tracks) |
| Validation data | 8 pairs (4 genres x 2 tracks) |
| Epochs | 5 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Total steps | 180 |
| GPU | Colab T4 (16 GB VRAM) |
| Training loss | 8.13 to 7.09 |
| Validation loss | 7.34 to 7.21 |

**Right side:** Placeholder for fine-tune loss curve plot.
Source: `finetune_history.json` (18 train loss points, 5 eval loss points).
If no image available, show a simple line chart with the data points above.

**Bottom note:**
This is a smoke test. A full run (800 tracks, 25 epochs) would produce
stronger genre specificity.

**Narration script:**
"We trained for 5 epochs on Colab, about 180 training steps. The training loss dropped from 8.1 to about 7.1. The validation loss went from 7.34 to 7.21, a modest decrease. The loss is high compared to the LSTM because we are predicting sequences of EnCodec tokens with a much larger codebook. This is a smoke test, not a full training run, but even this small run produced measurable improvements."

---

## Slide 13 — Task 4 Evaluation Approach

**Speaker:** D
**Time:** 14:00

**Title:** Task 4: Evaluation via Genre Consistency

**Layout:** Evaluation pipeline diagram

**Pipeline (horizontal flow):**

1. Generate 30s audio from text prompt (e.g., "electronic music with synthesizers")
2. Extract MFCC features from generated audio
3. Classify genre using SVM trained on real FMA data
4. Compare predicted genre to target genre

**Below diagram:**
- Metric: Genre classifier accuracy on generated audio
- Baseline: Pretrained MusicGen (no fine-tuning)
- Target: Fine-tuned MusicGen (our model)
- Test set: 1 sample per genre, 4 genres total

**Side note:**
Why not Frechet Audio Distance (FAD)?
Requires pretrained audio embedding model (VGGish or similar).
Not set up within project timeline. Genre accuracy is more interpretable.

**Narration script:**
"Evaluating continuous audio generation is harder than evaluating symbolic music because there is no simple ground truth to compare against. We use a genre consistency metric: does a classifier trained on real FMA audio correctly identify the genre of our generated audio? We trained an SVM classifier on MFCC features extracted from real FMA tracks. Then we generated one 30-second clip per genre from both the pretrained and fine-tuned models."

---

## Slide 14 — Task 4 Results and Discussion

**Speaker:** D
**Time:** 15:45

**Title:** Task 4: Results

**Layout:** Results on top, per-genre breakdown below, discussion at bottom

**Main result (large, centered):**
Genre Accuracy: 25% (pretrained) --> **75% (fine-tuned)**

**Per-genre breakdown table:**

| Genre | Prompt | Pretrained | Fine-tuned |
|-------|--------|------------|------------|
| Hip-Hop | "hip hop music with beats and rhythm" | Electronic (87%) | Electronic (97%) |
| Folk | "acoustic folk music with guitar" | Folk (100%) | Folk (100%) |
| Electronic | "electronic music with synthesizers" | Folk (100%) | Electronic (100%) |
| Rock | "energetic rock music with electric guitar" | Folk (100%) | Rock (100%) |

**Use green cells for correct predictions, red for incorrect.**

**Discussion bullets:**
- Pretrained model defaults to Folk-like acoustic textures for most prompts
- Fine-tuning teaches genre-specific audio characteristics
- Hip-Hop misclassified as Electronic makes sense: hip-hop production often uses synthesized beats
- Limitation: only 4 test samples (1 per genre), no confidence intervals

**Narration script:**
"The pretrained MusicGen scores 25% genre accuracy, 1 out of 4 correct. It only gets Folk right. The fine-tuned model scores 75%, 3 out of 4 correct. It correctly identifies Folk, Electronic, and Rock. The one miss is Hip-Hop, which gets classified as Electronic. The fine-tuned model's correct predictions all have confidence above 99.9%. A caveat: we only tested 4 samples, one per genre."

---

## Slide 15 — Related Work: Symbolic Generation

**Speaker:** D
**Time:** 16:30

**Title:** Related Work: Symbolic Music Generation

**Layout:** Timeline or table

| Year | Work | Model | Key Contribution | Relation to Ours |
|------|------|-------|-------------------|-----------------|
| 2005 | Allan & Williams | HMM | Classic baseline, BIC model selection | Our Markov chain is similar but simpler |
| 2017 | DeepBach (Hadjeres) | Gibbs + LSTM | Steerable generation, fix any voice | Explicit harmonic constraints we lack |
| 2017 | BachBot (Liang) | LSTM seq2seq | Turing test: fooled 1/3 listeners | Most similar architecture to ours |
| 2019 | Music Transformer (Huang) | Transformer + relative attention | Best perplexity (~1.2) on symbolic data | Shows headroom above our 1.97 |
| 2021 | BacHMMachine (Hahn) | Theory-guided HMM | Interpretable chord transitions | Music theory as model constraint |

**Bottom note:**
Our LSTM is closest to BachBot. We did not aim for SOTA perplexity.
Instead, we explored the tension between statistical fit and musical quality.

**Narration script:**
"For Task 1, the JSB Chorales are one of the oldest ML music benchmarks. DeepBach combined Gibbs sampling with LSTMs for steerable generation. BachBot used LSTM sequence-to-sequence models and ran a Turing test where one-third of listeners could not distinguish generated chorales from real Bach. The Music Transformer by Huang et al. achieved the best reported perplexity around 1.2, well below our 1.97. Our LSTM architecture is closest to BachBot's approach."

---

## Slide 16 — Related Work: Continuous Generation

**Speaker:** D
**Time:** 17:15

**Title:** Related Work: Continuous Audio Generation

**Layout:** Table

| Year | Work | Model | Key Contribution |
|------|------|-------|-------------------|
| 2017 | FMA Dataset (Defferrard) | - | 106K tracks, 161 genres, CC licensed benchmark |
| 2023 | MusicGen (Copet, NeurIPS) | Single-stage AR LM | Efficient codebook interleaving, our base model |
| 2023 | MusicLM (Agostinelli, Google) | Hierarchical audio LM | Semantic + acoustic token hierarchy |
| 2023 | AudioCraft (Meta) | Open-source framework | pip install audiocraft, democratized audio gen |
| 2025 | Genre Fine-tuning (IJCRT) | MusicGen-small fine-tuned | Same FMA genre approach as ours |

**Bottom note:**
Our contribution is the side-by-side comparison of symbolic and continuous
generation, showing how the same goal plays out across two representation spaces.

**Narration script:**
"For Task 4, MusicGen by Copet et al. is the foundation of our work. It introduced single-codebook interleaving for efficient autoregressive audio generation. Meta released MusicGen as part of AudioCraft. Most relevant to our approach is a 2025 IJCRT paper that fine-tuned MusicGen-small on FMA genres using essentially the same pipeline. Our contribution is the side-by-side comparison with the symbolic approach in Task 1."

---

## Slide 17 — Cross-Task Comparison

**Speaker:** A
**Time:** 18:00

**Title:** Symbolic vs Continuous: Two Paradigms Compared

**Layout:** Two-column comparison

| Dimension | Task 1: Symbolic | Task 4: Continuous |
|-----------|-----------------|-------------------|
| Representation | MIDI tokens (47-dim vocab) | Raw audio waveform (32kHz) |
| Model size | 1.1M parameters | 300M parameters |
| Training data | 368 chorales (~15 min audio equiv.) | 8,000 tracks (~67 hours) |
| Training time | Minutes (CPU) | Hours (GPU) |
| Evaluation | 5 precise metrics (perplexity, KL, voice-leading) | Genre classifier accuracy |
| Output quality | Structured but "mechanical" | Perceptually realistic |
| Interpretability | High (can inspect individual notes/voices) | Low (raw waveform) |
| Key tension | Statistical fit vs music theory rules | Generic quality vs genre specificity |

**Narration script:**
"Task 1 uses a compact, interpretable representation: 47 tokens, 1.1 million parameters, trains in minutes on a CPU. Task 4 uses a massive pretrained model: 300 million parameters, needs a GPU, and produces raw audio that sounds like real music. The symbolic approach gives us control and interpretability. The continuous approach gives us perceptual quality. Neither is strictly better. They serve different purposes."

---

## Slide 18 — Conclusion and Key Takeaways

**Speaker:** A
**Time:** 18:45

**Title:** Conclusion

**Layout:** Summary results + takeaways

**Results summary (two cards):**

Card 1 (red accent):
- Task 1: LSTM achieves **1.97 perplexity** on JSB Chorales
- 1.3x improvement over bigram Markov baseline
- But more parallel 5ths/octaves than baseline (18.1% vs 11.5%)

Card 2 (blue accent):
- Task 4: Fine-tuning MusicGen improves genre accuracy **25% to 75%**
- 3/4 genres correctly classified after fine-tuning
- Proof-of-concept with only 72 training samples

**Key takeaways (numbered list):**
1. Statistical objectives (cross-entropy) do not guarantee musical quality
2. Fine-tuning large pretrained models is practical and effective, even with limited data and compute
3. The choice between symbolic and continuous representations fundamentally shapes what you can evaluate

**Narration script:**
"In summary: for Task 1, our LSTM achieves 1.97 perplexity, a 1.3x improvement over a bigram baseline, but at the cost of more voice-leading violations. For Task 4, fine-tuning MusicGen improves genre classifier accuracy from 25% to 75%. Key takeaways: statistical objectives do not guarantee musical quality. Fine-tuning large pretrained models is practical even with limited data. And the choice between symbolic and continuous representations fundamentally shapes what questions you can ask about your model's output."

---

## Slide 19 — Thank You

**Speaker:** A
**Time:** 19:30

**Title:** Thank You

**Content (centered, minimal):**
Questions?

**Footer:**
Code and notebook: workbook.ipynb | workbook.html

**Narration script:**
"That concludes our 20-minute presentation. We will now play our generated music, which does not count toward the time limit per the assignment instructions."

---

## Slide 20 — Music Demo

**Speaker:** B (Task 1 audio), D (Task 4 audio)
**Time:** 20:00+

**Title:** Generated Music Demo

**Layout:** Two sections, file list for each task

**Task 1 — Symbolic (MIDI):**
1. Real Bach chorale (reference): `sample_chorale_real.mid`
2. Bigram Markov Chain output: `markov_chorale.mid`
3. LSTM output (temp 0.9): `symbolic_unconditioned.mid` (main deliverable)

**Task 4 — Continuous (Audio):**
4. Pretrained MusicGen, Electronic: `generated_audio/electronic_generated.mp3`
5. Fine-tuned MusicGen, Electronic: `generated_audio_finetuned/electronic_generated.mp3`
6. Pretrained MusicGen, Folk: `generated_audio/folk_generated.mp3`
7. Fine-tuned MusicGen, Folk: `generated_audio_finetuned/folk_generated.mp3`
8. Main deliverable: `continuous_conditioned.mp3` (Hip-Hop prompt)

**Note for builder:** This slide should have a clean list layout. No complex visuals needed,
the audio is played over the slide. Consider a waveform or equalizer animation as decoration.

**Narration script (Speaker B):**
"First, Task 1. Here is a real Bach chorale for reference. Now here is the Markov chain output. And here is our LSTM output at temperature 0.9."

**Narration script (Speaker D):**
"Now Task 4. First, the Electronic prompt, pretrained version, then fine-tuned. Next, Folk, pretrained then fine-tuned. Finally, our main deliverable, continuous_conditioned.mp3, generated with the prompt 'hip hop music with beats and rhythm.' Thank you for listening."

---

## Builder Notes

### Images to embed (if available in repo):
- `eda_piano_roll.png` (Slide 3 background/inset)
- `training_curves.png` (could reference on Slide 5 or show in notebook)
- `distribution_comparison.png` (optional, shown in notebook)
- `eval_summary.png` (could embed in Slide 7)
- `eval_task4_genre_accuracy.png` (could embed in Slide 14)
- `eda_musicgen_arch.png` (could embed in Slide 11)
- `eda_fma_overview.png` (could embed in Slide 9)
- `eda_fma_comparison.png` (could embed in Slide 10)

### Color scheme:
- Background: #1a1a2e or #0d1117 (dark)
- Text: #f0f0f0 (light)
- Task 1 accent: #e74c3c (red)
- Task 4 accent: #3498db (blue)
- Correct/positive: #2ecc71 (green)
- Incorrect/negative: #e74c3c (red)
- Neutral: #95a5a6 (gray)

### Font hierarchy:
- Slide title: 32-36pt, bold, white
- Section header: 24-28pt, semi-bold, accent color
- Body text: 18-22pt, regular, light gray
- Table text: 16-18pt
- Footer/caption: 14pt, dim gray
- Code: 16pt monospace

### Transitions:
- Slide transitions: fade (0.3s) or none
- No flying/spinning/bouncing animations
- Subtle opacity transitions for bullet reveals are acceptable

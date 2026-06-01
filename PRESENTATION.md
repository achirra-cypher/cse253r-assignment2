# CSE 253R Assignment 2 — Presentation Script & Flow
## Music Generation: Symbolic Unconditioned + Continuous Conditioned

**Format:** Hybrid recording. Slides for framing, context, and summaries. Jupyter notebook walkthrough for code, plots, and live outputs. Audio playback at the end.

**Total runtime:** 20 minutes of presentation content, plus music playback afterward (does not count toward 20 min per spec).

**Speakers:** 4 team members, 2 per task.

| Speaker | Task | Sections |
|---------|------|----------|
| **Speaker A** | Task 1 (Symbolic) | EDA, Data, Preprocessing |
| **Speaker B** | Task 1 (Symbolic) | Modeling, Evaluation, Related Work |
| **Speaker C** | Task 4 (Continuous) | EDA, Data, Preprocessing, Modeling |
| **Speaker D** | Task 4 (Continuous) | Evaluation, Related Work, Conclusion |

---

## Full Timeline

### 0:00 - 1:30 — Introduction (Speaker A)

**Show:** Slide 1 (Title), then Slide 2 (Overview)

**Speaker A narration:**

"Hi, we're [team names]. For this assignment we tackled two music generation tasks that sit on opposite ends of the representation spectrum. Task 1 is symbolic unconditioned generation, where we train models on Bach chorales represented as sequences of MIDI note tokens and generate new four-part harmony from scratch. Task 4 is continuous conditioned generation, where we fine-tune Meta's MusicGen model on the Free Music Archive to generate raw audio waveforms from text prompts like 'upbeat electronic music with synths.'

We picked this combination intentionally. The symbolic task gives us a small, controlled problem with interpretable evaluation metrics. We can check whether our generated music follows Bach's voice-leading rules. The continuous task gives us perceptually realistic output, actual listenable music, but requires working with a 300-million parameter pretrained model. The contrast between these two approaches is the throughline of our presentation.

Let me hand off to [Speaker A continues] for the data and preprocessing on Task 1."

**Transition cue:** Speaker A continues directly into Task 1 EDA.

---

### 1:30 - 3:45 — Task 1: EDA, Data Collection, Preprocessing (Speaker A)

**Show:** Slide 3 (JSB Chorales Dataset), then switch to notebook

#### 1:30 - 2:15 — Slide 3: Dataset Context

**Speaker A narration:**

"Our dataset for Task 1 is the JSB Chorales, 368 four-part harmonizations written by Johann Sebastian Bach. Each chorale is an SATB arrangement of a Lutheran hymn, meaning there are four voices: Soprano, Alto, Tenor, and Bass. These chorales are a canonical benchmark in symbolic music generation. They were used in DeepBach, BachBot, the Music Transformer paper, and several others. We load them directly from the music21 library's built-in corpus, so there's no external download step.

We represent each chorale as a time-by-four matrix of MIDI pitch values, quantized to a sixteenth-note grid. That means four steps per beat. The vocabulary is small: 47 tokens total, covering 46 unique MIDI pitches from MIDI 36 to 81, plus a rest token. We split the 368 chorales into 296 for training, 36 for validation, and 36 for testing."

#### 2:15 - 3:00 — Switch to Notebook: EDA Visualizations

**Show:** Notebook Cell 4 output (piano roll), Cell 6 output (pitch class distributions)

**Speaker A narration:**

"Let me show you the data in the notebook. This piano roll visualization shows one of the chorales. You can see the four voices color-coded: red is Soprano at the top, blue is Alto, green is Tenor, purple is Bass. The voices mostly move by step, which is typical of Bach's style. Voices rarely cross each other and they stay within well-defined ranges.

This next plot shows the pitch-class distribution for each voice. The Soprano range goes from MIDI 57 to 81, roughly middle C-sharp up to A above the staff. Bass covers MIDI 36 to 64. You can see each voice has a distinct pitch profile. The most common pitch classes are D, E, and G, reflecting that many of these chorales are in major keys."

**Show:** Notebook Cell 8 output (length, interval, pitch usage distributions)

**Speaker A narration:**

"Here are three more statistics. The chorale lengths cluster around 48 beats, which is roughly 12 bars. The interval distribution shows that the vast majority of melodic intervals are zero, meaning sustained notes, or stepwise motion of one to two semitones. Only about 2% of intervals are leaps larger than a sixth. That matters for evaluation later because it means even a simple model can get a lot of transitions right just by predicting 'same note again.'"

#### 3:00 - 3:45 — Notebook: Preprocessing Pipeline

**Show:** Notebook Cell 10 (Tokenizer class, make_sequences function)

**Speaker A narration:**

"For preprocessing, we built a Tokenizer class that maps between MIDI pitch space and a compact integer token space. The key function is make_sequences, which slices each chorale into overlapping windows of 64 timesteps, that is 16 beats or about 4 bars. We use a stride of 16 for training, giving us 75% overlap and 2,845 training sequences. Validation and test use stride 32. The roll_to_midi function at the bottom converts our token arrays back to playable MIDI files.

I'll hand off to [Speaker B] for the modeling."

**Transition cue:** Speaker B takes over.

---

### 3:45 - 7:00 — Task 1: Modeling (Speaker B)

**Show:** Slide 4 (Task 1 Modeling Overview), then notebook

#### 3:45 - 4:30 — Slide 4: Problem Formulation

**Speaker B narration:**

"The modeling problem is: given a sequence of four-voice tokens at time t, predict the four tokens at time t+1. We optimize cross-entropy loss summed across the four voices. This is a standard autoregressive next-token prediction setup.

We implemented two models. First, a Bigram Markov Chain as a baseline. This learns four independent 47-by-47 transition matrices, one per voice, with Laplace smoothing. It only looks at the current timestep. Second, a two-layer LSTM that processes all four voices jointly. The LSTM can capture longer-range dependencies through its hidden state, things like harmonic progressions that span multiple beats."

#### 4:30 - 5:15 — Notebook: Markov Chain

**Show:** Notebook Cell 13 (Markov chain fitting + transition matrix visualization)

**Speaker B narration:**

"Here is the Markov chain. We fit it by counting transitions in the training data. The transition matrix heatmaps show the probability of moving from each token to every other token. The bright diagonal tells you that by far the most common transition is staying on the same note. That is consistent with the interval distribution we saw earlier, where 77% of all transitions are unisons. The Markov chain achieves a test perplexity of 2.59, which is actually not bad for such a simple model, precisely because of that dominant diagonal."

#### 5:15 - 6:00 — Slide 5 (LSTM Architecture) + Notebook Cell 14

**Show:** Slide 5 (architecture diagram), then Cell 14 (architecture description)

**Speaker B narration:**

"Our LSTM takes the four voice tokens at each timestep through four separate embedding layers, each mapping the 47-token vocabulary into a 64-dimensional space. We concatenate those four embeddings into a 256-dimensional vector, feed it through a two-layer LSTM with hidden size 256 and dropout 0.3, and then project back out through four separate linear heads, one per voice. Total parameter count is about 1.1 million.

We train with teacher forcing: the model receives the ground-truth tokens at each step and predicts the next step. At generation time, we switch to autoregressive sampling, feeding the model's own predictions back as input. Temperature controls the randomness. We used temperature 0.9 for our main deliverable."

#### 6:00 - 7:00 — Notebook: Training + Results

**Show:** Notebook Cell 15 (training setup), Cell 16 (training curves), Cell 18 (perplexity comparison)

**Speaker B narration:**

"We trained for 28 epochs with early stopping, patience of 10. The best validation loss occurred at epoch 18. You can see in the training curves that validation loss plateaus around epoch 15 and starts to creep up, which is the early stopping trigger.

The LSTM achieves a test perplexity of 1.97, compared to 2.59 for the Markov chain. That is a 1.3x improvement. The improvement is modest, and we think the reason is that 77% of transitions are unisons. A bigram model already captures sustained notes well. The LSTM's advantage shows up on the remaining 23% of transitions where actual melodic motion happens. We will see this more clearly in the evaluation metrics.

Let me now walk through the evaluation."

**Transition cue:** Speaker B continues.

---

### 7:00 - 9:00 — Task 1: Evaluation (Speaker B)

**Show:** Slide 6 (Evaluation Metrics Overview), then notebook

#### 7:00 - 7:30 — Slide 6: Evaluation Framework

**Speaker B narration:**

"We evaluate with five metrics that capture different aspects of musical quality. Perplexity measures statistical fit. Pitch-class KL divergence checks whether generated music uses the same distribution of note classes as real Bach. Interval L1 measures whether the pattern of melodic jumps matches. Voice range violations count notes outside the historical SATB ranges. And parallel fifths and octaves count violations of a specific voice-leading rule from music theory, where two voices move in parallel by a perfect fifth or octave.

The key question is: does better perplexity translate to better music? We will see that the answer is nuanced."

#### 7:30 - 8:15 — Notebook: Evaluation Results

**Show:** Notebook Cell 32 (results table), Cell 33 (plots)

**Speaker B narration:**

"Here is the full comparison across 30 generated samples from each model. The LSTM wins on perplexity, 1.97 versus 2.59. Voice range violations are essentially zero for both models, with the LSTM slightly better at 0.01% versus 0.09%.

But look at the pitch-class KL divergence: the Markov chain actually scores slightly better, 0.176 versus 0.198. Same pattern for interval L1. And the LSTM has more parallel fifths and octaves, 18.1% versus 11.5%. Real Bach has only 3%.

These evaluation plots show it visually. The pitch-class distributions are close for both models. The interval histograms are similar. But the summary bar chart makes the tradeoff clear."

#### 8:15 - 9:00 — Notebook Cell 34 + Slide 7 (Discussion)

**Show:** Slide 7 (Task 1 Discussion)

**Speaker B narration:**

"So what is going on? The LSTM optimizes cross-entropy, which is a statistical objective. It learns that certain note transitions are more likely in Bach, but it does not have an explicit penalty for voice-leading violations. The Markov chain, being simpler, tends to produce blander output, it leans heavily on the dominant-diagonal pattern of sustaining notes, which accidentally avoids parallel motion because voices just sit still.

This is actually a well-known tension in music generation. The Music Transformer paper by Huang et al. achieves a test perplexity around 1.2 on similar data, much better than our 1.97. BachBot by Liang, which uses a comparable LSTM architecture, reports similar perplexity to ours. DeepBach by Hadjeres takes a different approach with Gibbs sampling, which explicitly enforces harmonic constraints. Our result is consistent with the literature: pure next-token prediction improves statistical fit but can worsen rule compliance without architectural constraints.

Now [Speaker C] will cover Task 4."

**Transition cue:** Speaker C takes over.

---

### 9:00 - 11:30 — Task 4: EDA, Data Collection, Preprocessing (Speaker C)

**Show:** Slide 8 (Task 4 Title), Slide 9 (FMA Dataset)

#### 9:00 - 9:45 — Slide 8-9: Task 4 Introduction and Dataset

**Speaker C narration:**

"Task 4 is continuous conditioned generation. The input is a text prompt describing a musical genre and style, and the output is a 30-second audio clip, actual playable music, not MIDI tokens.

Our dataset is FMA-small, the Free Music Archive. It contains 8,000 tracks, each 30 seconds long, evenly distributed across 8 genres: Hip-Hop, Pop, Folk, Experimental, Rock, International, Electronic, and Instrumental. That is 1,000 tracks per genre, perfectly balanced. The audio is mono, 22 kHz, and the whole dataset is about 67 hours of music. It is Creative Commons licensed and available on HuggingFace. The FMA dataset paper by Defferrard et al. from 2017 is the standard reference for music information retrieval benchmarks."

#### 9:45 - 10:30 — Notebook: FMA EDA

**Show:** Notebook Cell 23 (FMA overview plots), Cell 29 (genre-to-prompt mapping)

**Speaker C narration:**

"In the notebook you can see the genre distribution: perfectly balanced at 1,000 per genre. The audio specs show 30-second clips at 22 kHz mono. For fine-tuning MusicGen, we need to pair each audio clip with a text description. We created a genre-to-prompt mapping, shown here: 'hip hop music with beats and rhythm' for Hip-Hop, 'acoustic folk music with guitar' for Folk, and so on.

For our smoke test on Colab, we used a subset of 4 genres: Hip-Hop, Folk, Electronic, and Rock, with 20 tracks per genre. That gave us 72 training pairs and 8 validation pairs. This is a small training set, and we will come back to that when we discuss limitations."

#### 10:30 - 11:30 — Slide 10 (Why Fine-tune) + Notebook Cell 25

**Show:** Slide 10 (fine-tune vs scratch comparison), Notebook Cell 25 (comparison table)

**Speaker C narration:**

"A critical design decision for this task was whether to train a model from scratch or fine-tune a pretrained one. Training a continuous audio generation model from scratch, something like a spectrogram GAN or a WaveNet, would require hundreds of hours of audio and weeks of GPU time. With 6 days and a free Colab T4, that was not realistic. The output would almost certainly be noise.

Fine-tuning MusicGen gives us studio-quality audio out of the box because the pretrained model already knows how to produce coherent waveforms. Our job is to teach it genre-specific characteristics. This table in the notebook summarizes the tradeoff. The key point is that fine-tuning still counts as training our own weights, we are updating the transformer decoder parameters on our data, not just running inference on someone else's model."

**Transition cue:** Speaker C continues into modeling.

---

### 11:30 - 14:00 — Task 4: Modeling (Speaker C)

**Show:** Slide 11 (MusicGen Architecture), then notebook

#### 11:30 - 12:30 — Slide 11: MusicGen Architecture

**Speaker C narration:**

"MusicGen, published by Copet et al. at NeurIPS 2023, is a single-stage autoregressive language model that operates over discrete audio tokens. The architecture has three components. First, a frozen T5 text encoder that processes the input prompt into a sequence of embeddings. Second, a transformer decoder with 300 million parameters, this is the language model that generates audio token sequences conditioned on the text encoding. Third, a frozen EnCodec decoder that converts the discrete tokens back into a 32 kHz waveform.

The key insight is that only the transformer decoder gets fine-tuned. The text encoder and audio codec stay frozen. This means we are teaching the model to produce different token sequences for different genre prompts, while the audio reconstruction quality stays the same as the pretrained version."

#### 12:30 - 13:15 — Notebook: Architecture Diagram + Fine-tuning Code

**Show:** Notebook Cell 27 (architecture diagram), then reference to `musicgen_finetune.py`

**Speaker C narration:**

"This diagram in the notebook shows the pipeline visually. Text goes through T5, the transformer decoder generates EnCodec tokens, and EnCodec decodes to audio. The red box marks the fine-tuned component.

The fine-tuning code uses the HuggingFace Trainer API. We load the pretrained model from facebook/musicgen-small, prepare our FMA audio-text pairs, and train with standard cross-entropy loss on the EnCodec token sequences. The learning rate is 1e-4, lower than typical pretraining rates, to avoid catastrophic forgetting of the pretrained features. We use batch size 2 on a T4 GPU, which has 16 GB of VRAM."

#### 13:15 - 14:00 — Slide 12 (Training Details) + Training Curves

**Show:** Slide 12 (Training summary table + fine-tune loss curve)

**Speaker C narration:**

"We trained for 5 epochs on Colab, which took about 180 training steps. The training loss dropped from 8.1 to about 7.1 over those 5 epochs. The validation loss went from 7.34 to 7.21, a modest decrease. The loss is high compared to the LSTM because we are predicting sequences of EnCodec tokens, which have a much larger codebook. The absolute numbers are less meaningful than the relative improvement.

This is a smoke test, not a full training run. With the full 800-track dataset and 25 epochs, we would expect significantly better genre specificity. But even this small run produced measurable improvements, as we will see in the evaluation.

[Speaker D] will take it from here for Task 4 evaluation."

**Transition cue:** Speaker D takes over.

---

### 14:00 - 16:30 — Task 4: Evaluation (Speaker D)

**Show:** Slide 13 (Evaluation Approach), then notebook

#### 14:00 - 14:45 — Slide 13: Evaluation Framework

**Speaker D narration:**

"Evaluating continuous audio generation is harder than evaluating symbolic music because there is no simple ground truth to compare against. We cannot compute perplexity on a waveform the way we can on token sequences. Instead, we use a genre consistency metric: does a classifier trained on real FMA audio correctly identify the genre of our generated audio?

We trained an SVM classifier on MFCC features extracted from real FMA tracks, one per genre. Then we generated one 30-second clip per genre from both the pretrained MusicGen and our fine-tuned version, using the same text prompts. The classifier's accuracy on generated audio tells us how well each model captures genre-specific characteristics."

#### 14:45 - 15:45 — Notebook: Evaluation Results

**Show:** Notebook Cell 36 (results table), Cell 37 (bar chart + audio playback)

**Speaker D narration:**

"The results are striking. The pretrained MusicGen scores 25% genre accuracy, which is 1 out of 4 correct. It only gets Folk right. It classifies everything else as Folk, likely because its default generation style has acoustic qualities that overlap with the Folk category in our classifier.

The fine-tuned model scores 75%, 3 out of 4 correct. It correctly identifies Folk, Electronic, and Rock. The one miss is Hip-Hop, which gets classified as Electronic. Looking at the confidence scores, the fine-tuned model's correct predictions all have confidence above 99.9%. The Hip-Hop misclassification has 97% confidence for Electronic, suggesting the fine-tuned model's Hip-Hop output genuinely has strong electronic characteristics, which actually makes sense because hip-hop production often uses synthesized beats.

This bar chart shows the comparison visually: pretrained at 25% versus fine-tuned at 75%. A caveat: we only tested 4 samples, one per genre. A full evaluation with multiple samples per genre would give us confidence intervals, but the directional result is clear."

#### 15:45 - 16:30 — Discussion

**Show:** Slide 14 (Task 4 Discussion)

**Speaker D narration:**

"There are a few limitations to acknowledge. First, the training set was small, 72 tracks across 4 genres. A full run with 200 tracks per genre would likely produce better genre separation. Second, our evaluation used only 4 test samples. Third, we do not report Frechet Audio Distance, which is the standard metric in the audio generation literature, because computing it requires a pretrained audio embedding model that we did not set up in time.

Despite those caveats, the 25% to 75% jump demonstrates that fine-tuning on genre-specific data meaningfully changes the model's behavior. The pretrained model generates pleasant but generic audio. The fine-tuned model generates audio that a downstream classifier can reliably associate with the target genre."

---

### 16:30 - 18:00 — Related Work for Both Tasks (Speaker D)

**Show:** Slide 15 (Related Work, Task 1), Slide 16 (Related Work, Task 4)

#### 16:30 - 17:15 — Slide 15: Task 1 Related Work

**Speaker D narration:**

"For Task 1, the JSB Chorales are one of the oldest ML music benchmarks. Allan and Williams used hidden Markov models in 2005, establishing the baseline statistical approach. DeepBach by Hadjeres in 2017 combined Gibbs sampling with LSTMs to generate Bach-style chorales that are steerable, you can fix certain voices and let the model fill in the rest. BachBot by Liang used LSTM sequence-to-sequence models and ran a Turing test where one-third of listeners could not distinguish generated chorales from real Bach.

The Music Transformer by Huang et al. at ICLR 2019 introduced relative attention and achieved the best reported perplexity on symbolic music datasets, around 1.2, well below our 1.97. BacHMMachine by Hahn in 2021 used a theory-guided HMM with interpretable chord transitions.

Our work sits at the simpler end of this spectrum. Our LSTM architecture is closest to BachBot's approach. We did not aim for state-of-the-art perplexity. Instead, we used this task to explore the tension between statistical fit and music-theoretic quality, which we discussed in the evaluation section."

#### 17:15 - 18:00 — Slide 16: Task 4 Related Work

**Speaker D narration:**

"For Task 4, MusicGen by Copet et al. is the foundation of our work. It introduced single-codebook interleaving for efficient autoregressive audio generation, which was a major simplification over prior multi-stage approaches. MusicLM by Agostinelli at Google used a hierarchical approach with separate semantic and acoustic token streams.

Meta released MusicGen as part of AudioCraft, their open-source audio generation framework. The FMA dataset paper by Defferrard provides the standard benchmark for genre classification. Most relevant to our approach is a 2025 IJCRT paper that fine-tuned MusicGen-small on FMA genres using essentially the same pipeline we implemented. Their results showed similar genre accuracy improvements with fine-tuning.

Our contribution is not novelty in the MusicGen pipeline. Rather, it is the side-by-side comparison with the symbolic approach in Task 1, showing how the same fundamental goal, generating music, plays out across two very different representation spaces."

---

### 18:00 - 19:30 — Cross-Task Comparison and Conclusion (Speaker A)

**Show:** Slide 17 (Comparison Table), Slide 18 (Conclusion)

#### 18:00 - 18:45 — Slide 17: Comparison

**Speaker A narration:**

"To wrap up, let us compare what we learned from the two tasks. Task 1 uses a compact, interpretable representation: 47 tokens, 1.1 million parameters, trains in minutes on a CPU. We can evaluate it with precise metrics grounded in music theory. Task 4 uses a massive pretrained model: 300 million parameters, needs a GPU, and produces raw audio that sounds like real music. But evaluation is harder because there is no simple ground truth.

The symbolic approach gives us control and interpretability. We can measure specific voice-leading violations. The continuous approach gives us perceptual quality, the output sounds like something you would actually listen to. Neither approach is strictly better. They serve different purposes, and working with both gave us a more complete picture of the current state of music generation."

#### 18:45 - 19:30 — Slide 18: Conclusion

**Speaker A narration:**

"In summary: for Task 1, our LSTM achieves 1.97 perplexity on JSB Chorales, a 1.3x improvement over a bigram baseline, but at the cost of more voice-leading violations. For Task 4, fine-tuning MusicGen on FMA genres improves genre classifier accuracy from 25% to 75%.

Key takeaways: statistical objectives do not guarantee musical quality in the symbolic domain. Fine-tuning large pretrained models is practical and effective for audio generation, even with limited data and compute. And the choice between symbolic and continuous representations fundamentally shapes what questions you can ask about your model's output.

Now, as the spec suggests, we will play some of our generated music so you can hear it for yourselves."

---

### 19:30 - 20:00 — Buffer / Wrap-up (All)

**Show:** Slide 19 (Thank You / Questions)

**Speaker A narration:**

"That concludes our 20-minute presentation. We will now play our generated music, which does not count toward the time limit per the assignment instructions."

---

### 20:00+ — Music Playback (Does NOT count toward 20 min)

**Show:** Slide 20 (Music Demo), then play audio files

#### Task 1 Playback (Speaker B)

**Speaker B narration:**

"First, Task 1. Here is a real Bach chorale for reference."

**Play:** `sample_chorale_real.mid` (or pre-rendered audio version)

"Now here is the Markov chain output. Notice it stays on notes for a long time and the voice leading is somewhat random."

**Play:** `markov_chorale.mid`

"And here is our LSTM output at temperature 0.9. You should hear more varied melodic movement, and the voices interact more coherently, though you might catch some parallel motion that Bach would not have written."

**Play:** `symbolic_unconditioned.mid`

#### Task 4 Playback (Speaker D)

**Speaker D narration:**

"Now Task 4. We will compare pretrained versus fine-tuned MusicGen on two genres. First, the Electronic prompt. Here is the pretrained version."

**Play:** `generated_audio/electronic_generated.mp3` (~15 seconds)

"And here is the fine-tuned version of the same prompt."

**Play:** `generated_audio_finetuned/electronic_generated.mp3` (~15 seconds)

"You should hear that the fine-tuned version has more defined synthesizer textures and a clearer rhythmic pattern characteristic of electronic music."

"Next, the Folk prompt. Pretrained first."

**Play:** `generated_audio/folk_generated.mp3` (~15 seconds)

"And fine-tuned."

**Play:** `generated_audio_finetuned/folk_generated.mp3` (~15 seconds)

"Finally, here is our main deliverable, continuous_conditioned.mp3, generated with the prompt 'hip hop music with beats and rhythm.'"

**Play:** `continuous_conditioned.mp3`

"Thank you for listening."

---

## Notebook Cell Reference Map

For each presentation segment, which notebook cells to show:

| Time | Segment | Notebook Cells | What's Visible |
|------|---------|---------------|----------------|
| 2:15-3:00 | Task 1 EDA | Cell 4, Cell 6 | Piano roll, pitch class distributions |
| 3:00-3:45 | Task 1 Preprocessing | Cell 8, Cell 10 | Stats plots, Tokenizer class + make_sequences |
| 4:30-5:15 | Task 1 Markov | Cell 13 | Markov fit + transition matrix heatmaps |
| 5:15-6:00 | Task 1 LSTM | Cell 14, Cell 15 | Architecture description, training setup |
| 6:00-7:00 | Task 1 Training | Cell 16, Cell 18 | Training curves, perplexity comparison table |
| 7:30-8:15 | Task 1 Eval | Cell 32, Cell 33 | 5-metric results table, eval plots |
| 9:45-10:30 | Task 4 EDA | Cell 23, Cell 29 | FMA overview, genre-prompt mapping |
| 10:30-11:30 | Task 4 Why Fine-tune | Cell 25 | Comparison table |
| 12:30-13:15 | Task 4 Architecture | Cell 27 | MusicGen architecture diagram |
| 14:45-15:45 | Task 4 Eval | Cell 36, Cell 37 | Genre accuracy results, bar chart |

---

## Slide Deck Reference

| Slide # | Used At | Title |
|---------|---------|-------|
| 1 | 0:00 | Title Slide |
| 2 | 0:15 | Two Tasks Overview |
| 3 | 1:30 | JSB Chorales Dataset |
| 4 | 3:45 | Task 1: Modeling Overview |
| 5 | 5:15 | LSTM Architecture |
| 6 | 7:00 | Task 1: Evaluation Metrics |
| 7 | 8:15 | Task 1: Discussion + Related Work |
| 8 | 9:00 | Task 4: Continuous Conditioned Generation |
| 9 | 9:00 | FMA-Small Dataset |
| 10 | 10:30 | Fine-tuning vs Training from Scratch |
| 11 | 11:30 | MusicGen Architecture |
| 12 | 13:15 | Fine-tuning Training Details |
| 13 | 14:00 | Task 4: Evaluation Approach |
| 14 | 15:45 | Task 4: Discussion + Limitations |
| 15 | 16:30 | Related Work: Symbolic Generation |
| 16 | 17:15 | Related Work: Continuous Generation |
| 17 | 18:00 | Cross-Task Comparison |
| 18 | 18:45 | Conclusion + Key Takeaways |
| 19 | 19:30 | Thank You |
| 20 | 20:00+ | Music Demo |

---

## Artifact Paths (Corrected)

**IMPORTANT:** The folder `task1_weights_download/` is **misnamed**. It contains
Task 4 (MusicGen) fine-tuned weights, not Task 1 weights. Contents confirm this:
`config.json` has `model_type: "musicgen"`, and the folder holds `model-*.safetensors`
files (~2.3 GB each) plus T5/EnCodec configs.

| Artifact | Correct Path | Notes |
|----------|-------------|-------|
| Task 1 LSTM checkpoint | `lstm_checkpoint.pt` (repo root, gitignored) | ~4.5 MB, load via `torch.load()` |
| Task 1 preprocessed data | `X_train.npy`, `X_val.npy`, `X_test.npy`, `jsb_chorales.npy` (repo root, gitignored) | Regenerate with `python generate_data.py` |
| Task 1 training history | `training_history.json` (repo root) | 28 epochs, best at epoch 18 |
| Task 4 MusicGen weights | `task1_weights_download/` (MISNAMED, should be `finetuned_musicgen/`) | ~14 GB total with optimizer states |
| Task 4 fine-tune history | `finetune_history.json` (repo root) | 5 epochs, loss 8.1 to 7.1 |
| Task 4 pretrained samples | `generated_audio/` | 4 genre mp3s from base MusicGen |
| Task 4 fine-tuned samples | `generated_audio_finetuned/` | 4 genre mp3s from fine-tuned model |
| Task 4 main deliverable | `continuous_conditioned.mp3` (repo root) | Hip-Hop prompt, 30s |
| Task 1 main deliverable | `symbolic_unconditioned.mid` (repo root) | LSTM output, 192 steps |

**Action item:** Rename `task1_weights_download/` to `finetuned_musicgen/` before
any demo or code walkthrough that references checkpoint paths. The notebook and
`musicgen_generate.py` expect `--checkpoint finetuned_musicgen`.

---

## Recording Setup Notes

- **Software:** Zoom self-recording or OBS Studio. Export as MP4.
- **Screen share:** Full screen. Slides in one tab, notebook in another. Switch tabs when the script calls for it.
- **Speaker layout:** Gallery view or side-by-side webcam, so all speakers are visible.
- **Audio playback:** Share system audio when playing generated music. Test this before recording.
- **Upload:** Google Drive with "Anyone with the link" sharing. Paste URL into `video_url.txt`.
- **Verify:** Download your own video with `wget -O test.mp4 '<url>'` to confirm it works.

"""
models.py — Task 1: Symbolic Unconditioned Music Generation (JSB Chorales)

Contains:
    1. BigramMarkovChain   – per-voice bigram baseline with Laplace smoothing
    2. ChoraleLSTM          – 2-layer LSTM with per-voice embeddings and output heads
    3. train_lstm()         – training loop with early stopping
    4. generate_lstm()      – autoregressive sampling
    5. plot_training_curves(), compare_distributions() – visualisation helpers
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
# 1. Bigram Markov Chain
# ──────────────────────────────────────────────────────────────────────

class BigramMarkovChain:
    """
    Independent bigram (order-1 Markov) model for each of the 4 SATB voices.

    Parameters
    ----------
    vocab_size : int
        Number of tokens (default 47: 0 = rest/pad, 1-46 = MIDI 36-81).
    """

    def __init__(self, vocab_size: int = 47):
        self.vocab_size = vocab_size
        # transition_counts[v] is a (vocab_size, vocab_size) count matrix
        self.transition_counts = np.zeros((4, vocab_size, vocab_size), dtype=np.float64)
        # normalised probabilities (filled after fit)
        self.transition_probs = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: np.ndarray) -> "BigramMarkovChain":
        """
        Fit the bigram model from training windows.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, 64, 4)
            Tokenised chorale windows.

        Returns
        -------
        self
        """
        N, T, V = X_train.shape
        assert V == 4, f"Expected 4 voices, got {V}"

        self.transition_counts[:] = 0.0

        for voice in range(4):
            for n in range(N):
                for t in range(T - 1):
                    cur = int(X_train[n, t, voice])
                    nxt = int(X_train[n, t + 1, voice])
                    self.transition_counts[voice, cur, nxt] += 1

        # Laplace (add-1) smoothing and row normalisation
        smoothed = self.transition_counts + 1.0  # (4, V, V)
        row_sums = smoothed.sum(axis=-1, keepdims=True)  # (4, V, 1)
        self.transition_probs = smoothed / row_sums

        return self

    # ------------------------------------------------------------- generate
    def generate(
        self,
        length: int = 64,
        temperature: float = 1.0,
        seed_token: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Sample a chorale from the bigram model.

        Parameters
        ----------
        length : int
            Number of timesteps to generate.
        temperature : float
            Sampling temperature (>1 → more random, <1 → more peaked).
        seed_token : np.ndarray, shape (4,), optional
            Starting token for each voice.  If None, sampled uniformly.

        Returns
        -------
        np.ndarray, shape (length, 4)
            Generated token sequence.
        """
        assert self.transition_probs is not None, "Model not fitted yet."

        output = np.zeros((length, 4), dtype=np.int64)

        if seed_token is not None:
            output[0] = seed_token
        else:
            output[0] = np.random.randint(0, self.vocab_size, size=4)

        for voice in range(4):
            for t in range(1, length):
                cur = output[t - 1, voice]
                logits = np.log(self.transition_probs[voice, cur] + 1e-12)
                scaled = logits / temperature
                # numerically stable softmax
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
                output[t, voice] = np.random.choice(self.vocab_size, p=probs)

        return output

    # ----------------------------------------------------------- perplexity
    def perplexity(self, X_test: np.ndarray) -> float:
        """
        Compute per-token perplexity on held-out data.

        Parameters
        ----------
        X_test : np.ndarray, shape (N, T, 4)

        Returns
        -------
        float
            exp(average negative log-likelihood).
        """
        assert self.transition_probs is not None, "Model not fitted yet."

        N, T, V = X_test.shape
        total_nll = 0.0
        total_tokens = 0

        for voice in range(4):
            for n in range(N):
                for t in range(T - 1):
                    cur = int(X_test[n, t, voice])
                    nxt = int(X_test[n, t + 1, voice])
                    prob = self.transition_probs[voice, cur, nxt]
                    total_nll -= np.log(prob + 1e-12)
                    total_tokens += 1

        avg_nll = total_nll / total_tokens
        return float(np.exp(avg_nll))


# ──────────────────────────────────────────────────────────────────────
# 2. Chorale LSTM
# ──────────────────────────────────────────────────────────────────────

class ChoraleLSTM(nn.Module):
    """
    LSTM language model for 4-voice chorales.

    Architecture
    ------------
    - 4 separate Embedding layers (one per voice)
    - Concatenated embeddings → 2-layer LSTM
    - 4 separate Linear heads (one per voice) → logits over vocab

    Parameters
    ----------
    vocab_size : int
    embed_dim  : int   Per-voice embedding dimension.
    hidden_dim : int   LSTM hidden size.
    n_layers   : int   Number of stacked LSTM layers.
    dropout    : float Dropout between LSTM layers.
    """

    def __init__(
        self,
        vocab_size: int = 47,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Per-voice embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim) for _ in range(4)]
        )

        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=4 * embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Per-voice output heads
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, vocab_size) for _ in range(4)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.LongTensor, shape (batch, seq_len, 4)
            Token indices for the 4 voices.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, 4, vocab_size)
            Logits for each voice at each timestep.
        """
        # Embed each voice and concatenate
        embs = [self.embeddings[v](x[:, :, v]) for v in range(4)]  # list of (B, T, E)
        combined = torch.cat(embs, dim=-1)  # (B, T, 4*E)

        # LSTM
        lstm_out, _ = self.lstm(combined)  # (B, T, H)

        # Per-voice logits
        logits = torch.stack(
            [head(lstm_out) for head in self.heads], dim=2
        )  # (B, T, 4, vocab_size)

        return logits


# ──────────────────────────────────────────────────────────────────────
# 3. Training loop
# ──────────────────────────────────────────────────────────────────────

def train_lstm(
    model: ChoraleLSTM,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """
    Train the ChoraleLSTM with teacher forcing and early stopping.

    Teacher forcing setup:
        input  = X[:, :-1, :]   (tokens 0 … T-2)
        target = X[:, 1:, :]    (tokens 1 … T-1)

    Parameters
    ----------
    model     : ChoraleLSTM
    X_train   : np.ndarray, shape (N_train, 64, 4)
    X_val     : np.ndarray, shape (N_val, 64, 4)
    epochs    : int
    batch_size: int
    lr        : float
    device    : str

    Returns
    -------
    dict with keys 'train_loss', 'val_loss', 'val_perplexity'
        Each is a list of length = number of epochs actually run.
    """
    model = model.to(device)

    # Build DataLoaders
    def _make_loader(X, shuffle):
        inp = torch.from_numpy(X[:, :-1, :]).long()   # (N, 63, 4)
        tgt = torch.from_numpy(X[:, 1:, :]).long()    # (N, 63, 4)
        ds = TensorDataset(inp, tgt)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, shuffle=True)
    val_loader = _make_loader(X_val, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    history = {"train_loss": [], "val_loss": [], "val_perplexity": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        n_batches = 0

        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)

            logits = model(inp)  # (B, 63, 4, V)
            B, T, V_voices, V_vocab = logits.shape

            # Cross-entropy expects (N, C) logits and (N,) targets
            # Sum losses over the 4 voices
            loss = 0.0
            for v in range(4):
                loss += criterion(
                    logits[:, :, v, :].reshape(-1, V_vocab),
                    tgt[:, :, v].reshape(-1),
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / n_batches

        # ── Validate ──
        model.eval()
        val_running = 0.0
        val_batches = 0

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                B, T, V_voices, V_vocab = logits.shape

                loss = 0.0
                for v in range(4):
                    loss += criterion(
                        logits[:, :, v, :].reshape(-1, V_vocab),
                        tgt[:, :, v].reshape(-1),
                    )
                val_running += loss.item()
                val_batches += 1

        val_loss = val_running / val_batches
        # Perplexity: loss is average CE over 4 voices summed, so
        # per-voice average NLL = val_loss / 4.  But we report overall ppl.
        val_ppl = float(np.exp(val_loss / 4.0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.2f}"
        )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # save best weights
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ──────────────────────────────────────────────────────────────────────
# 4. Autoregressive Generation
# ──────────────────────────────────────────────────────────────────────

def generate_lstm(
    model: ChoraleLSTM,
    tokenizer_decode_fn=None,
    length: int = 64,
    temperature: float = 1.0,
    seed: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Autoregressively generate a chorale from the LSTM.

    Parameters
    ----------
    model              : ChoraleLSTM (already on `device`)
    tokenizer_decode_fn: callable or None
        If provided, maps token indices (np.ndarray) → MIDI pitches.
        Signature: decode(tokens: np.ndarray) -> np.ndarray.
        If None, raw token indices are returned.
    length             : int   Number of timesteps.
    temperature        : float Sampling temperature.
    seed               : np.ndarray, shape (1, 4) or (4,), optional
        Seed token(s) for the first timestep.  If None, sampled uniformly.
    device             : str

    Returns
    -------
    np.ndarray, shape (length, 4)
        Generated sequence (MIDI pitches if decode_fn given, else tokens).
    """
    model.eval()
    vocab_size = model.vocab_size

    # Seed
    if seed is not None:
        seed = np.asarray(seed).reshape(1, -1)  # (1, 4)
        assert seed.shape == (1, 4)
    else:
        seed = np.random.randint(0, vocab_size, size=(1, 4))

    generated = [seed[0]]  # list of (4,) arrays

    # Build initial input tensor: (1, 1, 4)
    current = torch.from_numpy(seed).long().unsqueeze(0).to(device)  # (1, 1, 4)

    hidden = None  # let LSTM track state across calls

    with torch.no_grad():
        for t in range(length - 1):
            # Embed & forward one step
            embs = [model.embeddings[v](current[:, :, v]) for v in range(4)]
            combined = torch.cat(embs, dim=-1)  # (1, 1, 4*E)

            if hidden is None:
                lstm_out, hidden = model.lstm(combined)
            else:
                lstm_out, hidden = model.lstm(combined, hidden)

            # Logits for this step
            next_tokens = np.zeros(4, dtype=np.int64)
            for v in range(4):
                logits_v = model.heads[v](lstm_out[:, -1, :])  # (1, V)
                logits_v = logits_v.squeeze(0) / temperature
                probs = F.softmax(logits_v, dim=-1)
                next_tokens[v] = torch.multinomial(probs, 1).item()

            generated.append(next_tokens)
            current = (
                torch.from_numpy(next_tokens)
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )  # (1, 1, 4)

    result = np.stack(generated, axis=0)  # (length, 4)

    if tokenizer_decode_fn is not None:
        result = tokenizer_decode_fn(result)

    return result


# ──────────────────────────────────────────────────────────────────────
# 5. Visualisation utilities
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> None:
    """
    Plot training / validation loss and perplexity curves side by side.

    Parameters
    ----------
    history : dict
        Must contain 'train_loss', 'val_loss', 'val_perplexity'.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="Val loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (sum of CE over 4 voices)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perplexity curve
    ax = axes[1]
    ax.plot(epochs, history["val_perplexity"], label="Val perplexity", color="tab:green", marker="^", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_distributions(
    real_data: np.ndarray,
    markov_data: np.ndarray,
    lstm_data: np.ndarray,
    vocab_size: int = 47,
) -> None:
    """
    Compare pitch-class histograms across real, Markov, and LSTM samples.

    Parameters
    ----------
    real_data   : np.ndarray  Flattened token array from real chorales.
    markov_data : np.ndarray  Flattened token array from Markov samples.
    lstm_data   : np.ndarray  Flattened token array from LSTM samples.
    vocab_size  : int
    """
    bins = np.arange(vocab_size + 1) - 0.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, data, title in zip(
        axes,
        [real_data, markov_data, lstm_data],
        ["Real Data", "Bigram Markov", "LSTM"],
    ):
        ax.hist(
            data.flatten(),
            bins=bins,
            density=True,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Token index")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.set_xlim(-0.5, vocab_size - 0.5)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Pitch Token Distribution Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

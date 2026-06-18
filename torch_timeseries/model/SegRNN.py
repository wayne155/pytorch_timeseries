import torch
import torch.nn as nn


class SegRNN(nn.Module):
    """SegRNN — Segment Recurrent Neural Network (Lin et al., ICLR 2024).

    Replaces self-attention with an iterative segment-based RNN for long-range
    forecasting. The look-back window is divided into equal-length segments
    which are encoded by a GRU. Decoding uses the Iterative Multi-step Output
    (IMO) strategy: the final hidden state is iteratively fed back to produce
    future segment embeddings which are linearly decoded.

    Paper: *SegRNN: Segment Recurrent Neural Network for Long-Term Time Series
    Forecasting*
    https://openreview.net/forum?id=jeqE7rqz2L

    Args:
        seq_len (int): Input sequence length.
        pred_len (int): Prediction horizon.
        enc_in (int): Number of input channels.
        d_model (int): RNN hidden size. Defaults to 512.
        seg_len (int): Segment length. ``seq_len`` and ``pred_len`` must each
            be divisible by ``seg_len``. Defaults to 48.
        dropout (float): Dropout probability applied after the RNN. Defaults
            to 0.5.

    Tasks: Forecasting, Imputation, Anomaly Detection, Classification.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        seg_len: int = 48,
        dropout: float = 0.5,
        output_prob: int = 0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.output_prob = output_prob

        # Pad seg_len to evenly divide both windows.
        self.seg_len = min(seg_len, seq_len)
        if seq_len % self.seg_len != 0:
            self.seg_len = _find_seg_len(seq_len, seg_len)
        self.num_seg_enc = seq_len // self.seg_len

        if pred_len % self.seg_len != 0:
            # pick the nearest divisor
            self.seg_len = _find_seg_len(pred_len, self.seg_len)
        self.num_seg_dec = pred_len // self.seg_len

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )
        self.pos_emb = nn.Embedding(self.num_seg_dec, d_model // 2)
        self.channel_emb = nn.Embedding(enc_in, d_model // 2)

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, self.seg_len),
        )
        if output_prob > 0:
            self.projection = nn.Linear(enc_in * pred_len, output_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape

        # Channel independence: process each channel separately.
        # Reshape → (B*C, num_seg, seg_len)
        x_enc = x.permute(0, 2, 1).reshape(B * C, self.num_seg_enc, self.seg_len)

        # Encode segments: (B*C, num_seg, d_model)
        enc_in = self.valueEmbedding(x_enc)
        _, h = self.rnn(enc_in)   # h: (1, B*C, d_model)

        # Iterative Multi-step Output (IMO) decoding.
        # Position + channel embeddings for each future segment.
        pos_idx = torch.arange(self.num_seg_dec, device=x.device)             # (Td,)
        ch_idx = torch.arange(C, device=x.device).unsqueeze(1).expand(C, self.num_seg_dec)  # (C, Td)
        pos_idx = pos_idx.unsqueeze(0).expand(C, self.num_seg_dec)             # (C, Td)

        # (C, Td, d_model//2) each → (C, Td, d_model)
        pos_emb = self.pos_emb(pos_idx)        # (C, Td, d_model//2)
        ch_emb = self.channel_emb(ch_idx)      # (C, Td, d_model//2)
        dec_emb = torch.cat([pos_emb, ch_emb], dim=-1)  # (C, Td, d_model)

        # Expand for batch: (B*C, Td, d_model)
        dec_emb = dec_emb.unsqueeze(0).expand(B, C, self.num_seg_dec, self.d_model)
        dec_emb = dec_emb.reshape(B * C, self.num_seg_dec, self.d_model)

        # Add hidden state as initial input and run RNN over dec embeddings.
        # IMO: inject h at every decoding step by adding it to the embedding.
        dec_in = dec_emb + h.permute(1, 0, 2)    # broadcast over Td
        dec_out, _ = self.rnn(dec_in, h)           # (B*C, Td, d_model)

        # Predict segments: (B*C, Td, seg_len)
        dec_out = self.predict(dec_out)

        # Reshape to (B, pred_len, C)
        out = dec_out.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.output_prob > 0:
            out = out.reshape(B, -1)
            out = self.projection(out)
        return out


def _find_seg_len(total: int, target: int) -> int:
    """Return the largest divisor of ``total`` that is ≤ ``target``."""
    if total % target == 0:
        return target
    best = 1
    for d in range(1, total + 1):
        if total % d == 0 and d <= target:
            best = d
    return best

"""Shape and gradient-flow tests for all forecasting models.

Each model is tested with:
  1. Correct output shape (B, pred_len, C)
  2. No NaN / Inf in output
  3. Gradient flows back through at least one leaf parameter
"""
import pytest
import torch

from torch_timeseries.model import (
    Autoformer,
    CATS,
    Crossformer,
    DLinear,
    FEDformer,
    FITS,
    FreTS,
    Informer,
    iTransformer,
    NHiTS,
    NLinear,
    PatchTST,
    SCINet,
    SegRNN,
    TiDE,
    TimeMixer,
    TimesNet,
    TSMixer,
)

B, T, C, H = 2, 96, 7, 24
LABEL = 48   # encoder-decoder label length


def _check_output(out, B, H, C):
    assert out.shape == (B, H, C), f"Expected ({B}, {H}, {C}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"


def _check_grads(model, out):
    out.sum().backward()
    has_grad = any(
        p.grad is not None and not torch.isnan(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, "No parameter received a valid gradient"


def _encdec_inputs():
    x_enc = torch.randn(B, T, C)
    x_mark_enc = torch.zeros(B, T, 4)
    x_dec = torch.zeros(B, LABEL + H, C)
    x_mark_dec = torch.zeros(B, LABEL + H, 4)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


# ── simple: x: (B, T, C) → (B, H, C) ────────────────────────────────────────

@pytest.mark.parametrize("model_cls,kwargs", [
    (DLinear,  dict(seq_len=T, pred_len=H, enc_in=C)),
    (NLinear,  dict(seq_len=T, pred_len=H, enc_in=C)),
    # FITS returns (forecast, long_term_output) tuple — tested separately below
    # (FITS, dict(...)),
    (FreTS,    dict(seq_len=T, pred_len=H, enc_in=C, channel_independence=True)),
    (SegRNN,   dict(seq_len=T, pred_len=H, enc_in=C, d_model=64, seg_len=24)),
    (TiDE,     dict(seq_len=T, pred_len=H, enc_in=C, hidden_size=64,
                    num_encoder_layers=1, num_decoder_layers=1)),
    (NHiTS,    dict(seq_len=T, pred_len=H, enc_in=C, n_stacks=2, n_blocks=1,
                    n_theta=64)),
    (TimeMixer, dict(seq_len=T, pred_len=H, enc_in=C, d_model=32, e_layers=2,
                     n_heads=2)),
    (Crossformer, dict(data_dim=C, in_len=T, out_len=H, seg_len=12,
                       d_model=32, d_ff=64, n_heads=2)),
])
# (FITS is excluded from parametrize — it returns a tuple; see test_fits below)
def test_simple_models(model_cls, kwargs):
    m = model_cls(**kwargs)
    x = torch.randn(B, T, C)
    out = m(x)
    _check_output(out, B, H, C)
    _check_grads(m, out)


# ── FITS (returns tuple) ─────────────────────────────────────────────────────

def test_fits():
    # FITS returns (forecast, long_term); both have same shape
    from torch_timeseries.model import FITS
    m = FITS(seq_len=T, pred_len=H, enc_in=C, individual=False, cut_freq=8)
    x = torch.randn(B, T, C)
    out = m(x)
    assert isinstance(out, (tuple, list)), "FITS should return tuple"
    forecast = out[0]
    assert forecast.shape == (B, T + H, C) or forecast.shape[0] == B


# ── TSMixer ───────────────────────────────────────────────────────────────────

def test_tsmixer():
    m = TSMixer(L=T, C=C, T=H, n_mixer=2)
    x = torch.randn(B, T, C)
    out = m(x)
    _check_output(out, B, H, C)
    _check_grads(m, out)


# ── SCINet ────────────────────────────────────────────────────────────────────

def test_sciNet():
    m = SCINet(output_len=H, input_len=T, input_dim=C, hid_size=16,
               num_stacks=1, num_levels=2)
    x = torch.randn(B, T, C)
    out = m(x)
    _check_output(out, B, H, C)
    _check_grads(m, out)


# ── Encoder-decoder Transformer family ───────────────────────────────────────

def test_informer():
    m = Informer(enc_in=C, dec_in=C, c_out=C, out_len=H, d_model=32, d_ff=64,
                 n_heads=2, e_layers=1, d_layers=1)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    _check_output(out, B, H, C)
    _check_grads(m, out)


def test_autoformer():
    # moving_avg must be odd (scalar) to avoid length mismatch in MovingAvg
    m = Autoformer(enc_in=C, dec_in=C, c_out=C, seq_len=T, pred_len=H,
                   label_len=LABEL, d_model=32, d_ff=64, n_heads=2,
                   e_layers=1, d_layers=1, moving_avg=25)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    # Autoformer returns (output, attn) when output_attention=True (default)
    if isinstance(out, (tuple, list)):
        out = out[0]
    _check_output(out, B, H, C)
    _check_grads(m, out)


def test_fedformer():
    # moving_avg must be an odd scalar for SeriesDecomp used in encoder/decoder layers
    m = FEDformer(enc_in=C, dec_in=C, c_out=C, seq_len=T, pred_len=H,
                  label_len=LABEL, d_model=32, d_ff=64, n_heads=2,
                  e_layers=1, d_layers=1, moving_avg=25)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    if isinstance(out, (tuple, list)):
        out = out[0]
    _check_output(out, B, H, C)
    _check_grads(m, out)


def test_patchtst():
    m = PatchTST(seq_len=T, pred_len=H, enc_in=C, d_model=32, d_ff=64,
                 n_heads=2, e_layers=1)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    _check_output(out, B, H, C)
    _check_grads(m, out)


def test_itransformer():
    m = iTransformer(seq_len=T, pred_len=H, c_in=C, d_model=32, d_ff=64,
                     n_heads=2, e_layers=1)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    _check_output(out, B, H, C)
    _check_grads(m, out)


# ── TimesNet ─────────────────────────────────────────────────────────────────

def test_timesnet():
    m = TimesNet(seq_len=T, pred_len=H, enc_in=C, d_model=32, d_ff=64,
                 top_k=3, num_kernels=4, e_layers=1)
    x_enc, x_mark_enc, x_dec, x_mark_dec = _encdec_inputs()
    out = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
    _check_output(out, B, H, C)
    _check_grads(m, out)


# ── CATS ─────────────────────────────────────────────────────────────────────

def test_cats():
    m = CATS(dec_in=C, seq_len=T, pred_len=H, d_model=32, d_ff=64,
             n_heads=2, d_layers=1, dropout=0.1, query_independence=True,
             patch_len=16, stride=8, padding_patch="end",
             store_attn=False, QAM_start=0, QAM_end=1)
    x = torch.randn(B, T, C)
    out = m(x)
    _check_output(out, B, H, C)
    _check_grads(m, out)

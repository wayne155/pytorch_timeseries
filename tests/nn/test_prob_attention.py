"""Tests for ProbAttention (Informer's sparse attention mechanism).

ProbAttention uses random key sampling + sparsity measurement to select the
top-U queries, giving O(L log L) complexity vs O(L²) for FullAttention.

Output format: context is (B, L, H, E) — same layout as inputs.
Attention weights are always returned with shape (B, H, L, L).
"""
import torch
import pytest

from torch_timeseries.nn import ProbAttention, AttentionLayer


B, L, H, E = 4, 32, 4, 16   # batch, seq_len, heads, head_dim


def _qkv(b=B, l=L, h=H, e=E):
    """Return (B, L, H, E) query / key / value tensors."""
    return (
        torch.randn(b, l, h, e),
        torch.randn(b, l, h, e),
        torch.randn(b, l, h, e),
    )


# ── construction ─────────────────────────────────────────────────────────────


class TestProbAttentionConstruction:
    def test_default_params(self):
        attn = ProbAttention()
        assert attn.factor == 5
        assert attn.mask_flag is True

    def test_custom_factor(self):
        attn = ProbAttention(factor=3)
        assert attn.factor == 3

    def test_output_attention_flag_stored(self):
        attn = ProbAttention(output_attention=True)
        assert attn.output_attention

    def test_no_mask_flag(self):
        attn = ProbAttention(mask_flag=False)
        assert not attn.mask_flag


# ── output shape ─────────────────────────────────────────────────────────────


class TestProbAttentionShape:
    def test_context_shape_no_mask(self):
        """Context returned in (B, L, H, E) layout — same as input."""
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv()
        ctx, weights = attn(q, k, v, attn_mask=None)
        assert ctx.shape == (B, L, H, E)

    def test_context_shape_with_mask(self):
        """Causal mask (mask_flag=True) requires L_Q == L_K."""
        attn = ProbAttention(mask_flag=True)
        q, k, v = _qkv()
        ctx, weights = attn(q, k, v, attn_mask=None)
        assert ctx.shape == (B, L, H, E)

    def test_attention_weights_shape(self):
        """Attention weights always returned with shape (B, H, L, L)."""
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv()
        _, weights = attn(q, k, v, attn_mask=None)
        assert weights is not None
        assert weights.shape == (B, H, L, L)

    def test_short_sequence(self):
        """L=4 (< typical factor*log(L)) — should not crash."""
        attn = ProbAttention(mask_flag=False, factor=1)
        q, k, v = _qkv(b=2, l=4)
        ctx, _ = attn(q, k, v, attn_mask=None)
        assert ctx.shape == (2, 4, H, E)

    def test_different_factor_same_shape(self):
        for factor in (1, 3, 5):
            attn = ProbAttention(mask_flag=False, factor=factor)
            q, k, v = _qkv()
            ctx, _ = attn(q, k, v, attn_mask=None)
            assert ctx.shape == (B, L, H, E)

    def test_batch_size_one(self):
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv(b=1)
        ctx, _ = attn(q, k, v, attn_mask=None)
        assert ctx.shape == (1, L, H, E)


# ── numerical behaviour ───────────────────────────────────────────────────────


class TestProbAttentionNumerics:
    def test_output_finite_no_mask(self):
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv()
        ctx, _ = attn(q, k, v, attn_mask=None)
        assert torch.isfinite(ctx).all()

    def test_output_finite_causal(self):
        attn = ProbAttention(mask_flag=True)
        q, k, v = _qkv()
        ctx, _ = attn(q, k, v, attn_mask=None)
        assert torch.isfinite(ctx).all()

    def test_weights_sum_to_one(self):
        """Rows of attention weights should sum to ~1 (they're softmaxed)."""
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv()
        _, weights = attn(q, k, v, attn_mask=None)
        # Rows selected by ProbAttn sum to 1; non-selected rows are uniform 1/L
        row_sums = weights.sum(dim=-1)  # (B, H, L)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_output_contiguous(self):
        attn = ProbAttention(mask_flag=False)
        q, k, v = _qkv()
        ctx, _ = attn(q, k, v, attn_mask=None)
        assert ctx.is_contiguous()


# ── gradient flow ─────────────────────────────────────────────────────────────


class TestProbAttentionGradients:
    def test_gradient_through_values(self):
        attn = ProbAttention(mask_flag=False)
        q, k = torch.randn(B, L, H, E), torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E, requires_grad=True)
        ctx, _ = attn(q, k, v, attn_mask=None)
        ctx.sum().backward()
        assert v.grad is not None

    def test_gradient_finite(self):
        attn = ProbAttention(mask_flag=False)
        q, k = torch.randn(B, L, H, E), torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E, requires_grad=True)
        ctx, _ = attn(q, k, v, attn_mask=None)
        ctx.sum().backward()
        assert torch.isfinite(v.grad).all()


# ── AttentionLayer wrapping ────────────────────────────────────────────────────


class TestProbAttentionLayer:
    """ProbAttention used inside the AttentionLayer wrapper (as Informer does)."""

    def _build(self, d_model=64, n_heads=4):
        return AttentionLayer(
            attention=ProbAttention(mask_flag=False),
            d_model=d_model,
            n_heads=n_heads,
        )

    def test_output_shape(self):
        layer = self._build()
        x = torch.randn(B, L, 64)
        out, attn = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, 64)

    def test_gradient_flows(self):
        layer = self._build()
        x = torch.randn(B, L, 64, requires_grad=True)
        out, _ = layer(x, x, x, attn_mask=None)
        out.sum().backward()
        assert x.grad is not None

    def test_no_nan_output(self):
        layer = self._build()
        x = torch.randn(B, L, 64)
        out, _ = layer(x, x, x, attn_mask=None)
        assert torch.isfinite(out).all()

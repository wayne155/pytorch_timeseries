"""Tests for MultiWaveletTransform and MultiWaveletCross."""
import pytest
import torch

from torch_timeseries.nn import MultiWaveletCross, MultiWaveletTransform


# Shared small dimensions that keep tests fast.
B, L, H, E = 2, 16, 2, 4   # sequence len must be power-of-2 for wavelet depth
ICH = H * E                  # flattened channel dim expected by both modules


# ---------------------------------------------------------------------------
# MultiWaveletTransform
# ---------------------------------------------------------------------------

class TestMultiWaveletTransform:
    """Tests for the 1D multi-wavelet self-attention block."""

    def _build(self, k=4, alpha=8, c=8, nCZ=1, L=0, base="legendre"):
        return MultiWaveletTransform(ich=ICH, k=k, alpha=alpha, c=c,
                                     nCZ=nCZ, L=L, base=base)

    def _qkv(self, seq_len=L):
        t = torch.randn(B, seq_len, H, E)
        return t, t.clone(), t.clone()

    # --- construction ---

    def test_construction_defaults(self):
        mwt = MultiWaveletTransform(ich=ICH)
        assert isinstance(mwt, torch.nn.Module)

    def test_construction_legendre(self):
        mwt = self._build(base="legendre")
        assert mwt.k == 4

    def test_construction_chebyshev(self):
        mwt = self._build(base="chebyshev")
        assert mwt.k == 4

    def test_construction_multi_nCZ(self):
        mwt = self._build(nCZ=2)
        assert len(mwt.MWT_CZ) == 2

    # --- forward shape ---

    def test_output_is_tuple(self):
        mwt = self._build()
        q, k, v = self._qkv()
        out = mwt(q, k, v, attn_mask=None)
        assert isinstance(out, tuple) and len(out) == 2

    def test_output_attention_is_none(self):
        mwt = self._build()
        q, k, v = self._qkv()
        _, attn = mwt(q, k, v, attn_mask=None)
        assert attn is None

    def test_output_value_shape_matches_input(self):
        mwt = self._build()
        q, k, v = self._qkv()
        V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L, H, E)

    def test_longer_queries_than_values(self):
        """When L_q > L_v the module zero-pads values to match."""
        mwt = self._build()
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L // 2, H, E)
        v = torch.randn(B, L // 2, H, E)
        V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L, H, E)

    def test_shorter_queries_than_values(self):
        """When L_q < L_v values are truncated to match."""
        mwt = self._build()
        q = torch.randn(B, L // 2, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L // 2, H, E)

    def test_output_is_finite(self):
        mwt = self._build()
        q, k, v = self._qkv()
        V, _ = mwt(q, k, v, attn_mask=None)
        assert torch.isfinite(V).all()

    def test_different_batch_sizes(self):
        mwt = self._build()
        for b in (1, 4):
            q = torch.randn(b, L, H, E)
            V, _ = mwt(q, q.clone(), q.clone(), attn_mask=None)
            assert V.shape[0] == b

    def test_nCZ_two_layers(self):
        mwt = self._build(nCZ=2)
        q, k, v = self._qkv()
        V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L, H, E)

    def test_chebyshev_forward(self):
        mwt = self._build(base="chebyshev")
        q, k, v = self._qkv()
        V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L, H, E)

    def test_no_grad_forward(self):
        mwt = self._build()
        q, k, v = self._qkv()
        with torch.no_grad():
            V, _ = mwt(q, k, v, attn_mask=None)
        assert V.shape == (B, L, H, E)

    def test_gradient_flows(self):
        # queries/keys are only used for shape; grad flows through values.
        mwt = self._build()
        q = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E, requires_grad=True)
        V, _ = mwt(q, q, v, attn_mask=None)
        V.sum().backward()
        assert v.grad is not None


# ---------------------------------------------------------------------------
# MultiWaveletCross
# ---------------------------------------------------------------------------

class TestMultiWaveletCross:
    """Tests for the 1D multi-wavelet cross-attention layer."""

    C, K = 4, 4          # internal wavelet dimensions

    def _build(self, seq_len_q=L, seq_len_kv=L, modes=4,
               base="legendre", activation="tanh"):
        return MultiWaveletCross(
            in_channels=self.C,
            out_channels=self.C,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            modes=modes,
            c=self.C,
            k=self.K,
            ich=ICH,
            L=0,
            base=base,
            activation=activation,
        )

    def _qkv(self, seq_len=L):
        t = torch.randn(B, seq_len, H, E)
        return t, t.clone(), t.clone()

    # --- construction ---

    def test_construction_legendre(self):
        mwc = self._build(base="legendre")
        assert isinstance(mwc, torch.nn.Module)

    def test_construction_chebyshev(self):
        mwc = self._build(base="chebyshev")
        assert isinstance(mwc, torch.nn.Module)

    def test_construction_tanh_activation(self):
        mwc = self._build(activation="tanh")
        assert mwc.attn1.activation == "tanh"

    def test_construction_softmax_activation(self):
        mwc = self._build(activation="softmax")
        assert mwc.attn1.activation == "softmax"

    # --- forward shape ---

    def test_output_is_tuple(self):
        mwc = self._build()
        q, k, v = self._qkv()
        out = mwc(q, k, v)
        assert isinstance(out, tuple) and len(out) == 2

    def test_output_attention_is_none(self):
        mwc = self._build()
        q, k, v = self._qkv()
        _, attn = mwc(q, k, v)
        assert attn is None

    def test_output_value_shape(self):
        mwc = self._build()
        q, k, v = self._qkv()
        V, _ = mwc(q, k, v)
        assert V.shape == (B, L, ICH)

    def test_output_is_finite(self):
        mwc = self._build()
        q, k, v = self._qkv()
        V, _ = mwc(q, k, v)
        assert torch.isfinite(V).all()

    def test_with_mask_kwarg(self):
        mwc = self._build()
        q, k, v = self._qkv()
        V, _ = mwc(q, k, v, mask=None)
        assert V.shape == (B, L, ICH)

    def test_different_batch_sizes(self):
        mwc = self._build()
        for b in (1, 4):
            q = torch.randn(b, L, H, E)
            V, _ = mwc(q, q.clone(), q.clone())
            assert V.shape[0] == b

    def test_no_grad_forward(self):
        mwc = self._build()
        q, k, v = self._qkv()
        with torch.no_grad():
            V, _ = mwc(q, k, v)
        assert V.shape == (B, L, ICH)

    def test_gradient_flows(self):
        mwc = self._build()
        q = torch.randn(B, L, H, E, requires_grad=True)
        v = torch.randn(B, L, H, E)
        V, _ = mwc(q, q, v)
        V.sum().backward()
        assert q.grad is not None

    def test_chebyshev_forward(self):
        mwc = self._build(base="chebyshev")
        q, k, v = self._qkv()
        V, _ = mwc(q, k, v)
        assert V.shape == (B, L, ICH)

    def test_softmax_activation_forward(self):
        mwc = self._build(activation="softmax")
        q, k, v = self._qkv()
        V, _ = mwc(q, k, v)
        assert V.shape == (B, L, ICH)

    def test_longer_query_sequence(self):
        """Query longer than key/value — module pads internally."""
        mwc = self._build(seq_len_q=32, seq_len_kv=16)
        q = torch.randn(B, 32, H, E)
        k = torch.randn(B, 16, H, E)
        v = torch.randn(B, 16, H, E)
        V, _ = mwc(q, k, v)
        assert V.shape == (B, 32, ICH)

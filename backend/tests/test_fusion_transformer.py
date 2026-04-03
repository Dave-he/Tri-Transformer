import pytest
import torch

from app.model.fusion_transformer import (
    FusionBlockConfig,
    FusionTransformer,
    GroupedMaskMultiHeadAttention,
    InputFusion,
    ModeEmbedding,
    GenerationMode,
    FusionBlock,
)


@pytest.fixture
def small_config():
    return FusionBlockConfig(
        d_model=128,
        num_heads_total=12,
        num_kv_heads=2,
        intermediate_size=256,
        num_layers=2,
        dropout=0.0,
        vocab_size=256,
    )


@pytest.fixture
def model(small_config):
    return FusionTransformer(
        vocab_size=small_config.vocab_size,
        d_model=small_config.d_model,
        num_heads_total=small_config.num_heads_total,
        num_kv_heads=small_config.num_kv_heads,
        num_layers=small_config.num_layers,
        intermediate_size=small_config.intermediate_size,
        dropout=small_config.dropout,
    )


class TestGroupedMaskMultiHeadAttention:
    def test_initialization(self, small_config):
        attn = GroupedMaskMultiHeadAttention(
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
        )
        assert attn.h_group == small_config.num_heads_total // 3
        assert attn.head_dim == small_config.d_model // small_config.num_heads_total

    def test_forward_mask_isolation(self, small_config):
        attn = GroupedMaskMultiHeadAttention(
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
        )
        B, S, D = 2, 8, small_config.d_model
        x = torch.randn(B, S, D)

        mask_f, mask_b, mask_d = self._create_masks(S, x.device)

        out = attn(x, mask_f, mask_b, mask_d)
        assert out.shape == (B, S, D)

    def _create_masks(self, seq_len, device):
        mask_f = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask_f = mask_f.masked_fill(mask_f == 0, float("-inf")).masked_fill(mask_f == 1, 0)

        mask_b = torch.triu(torch.ones(seq_len, seq_len, device=device))
        mask_b = mask_b.masked_fill(mask_b == 0, float("-inf")).masked_fill(mask_b == 1, 0)

        mask_d = torch.zeros(seq_len, seq_len, device=device)
        return mask_f, mask_b, mask_d


class TestInputFusion:
    def test_initialization(self):
        fusion = InputFusion(d_model=128)
        assert fusion.d_model == 128

    def test_default_timestep_for_ar(self):
        fusion = InputFusion(d_model=128)
        x_core = torch.randn(2, 8, 128)
        out = fusion(x_core, timesteps=None, mode=GenerationMode.FORWARD_AR)
        assert out.shape == x_core.shape

    def test_diffusion_timestep(self):
        fusion = InputFusion(d_model=128)
        x_core = torch.randn(2, 8, 128)
        timesteps = torch.randint(0, 1000, (2,))
        out = fusion(x_core, timesteps=timesteps, mode=GenerationMode.DIFFUSION)
        assert out.shape == x_core.shape


class TestModeEmbedding:
    def test_embed_forward_ar(self):
        embed = ModeEmbedding(d_model=128)
        out = embed(GenerationMode.FORWARD_AR)
        assert out.shape == (128,)

    def test_embed_all_modes(self):
        embed = ModeEmbedding(d_model=128)
        modes = [
            GenerationMode.FORWARD_AR,
            GenerationMode.BACKWARD_AR,
            GenerationMode.DIFFUSION,
        ]
        for mode in modes:
            out = embed(mode)
            assert out.shape == (128,)


class TestFusionBlock:
    def test_initialization(self, small_config):
        block = FusionBlock(
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            intermediate_size=small_config.intermediate_size,
        )
        assert block is not None

    def test_forward_with_masks(self, small_config):
        block = FusionBlock(
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            intermediate_size=small_config.intermediate_size,
        )
        B, S, D = 2, 8, small_config.d_model
        x = torch.randn(B, S, D)
        c = torch.randn(B, D)

        mask_f, mask_b, mask_d = self._create_masks(S, x.device)

        out = block(x, c, mask_f, mask_b, mask_d)
        assert out.shape == x.shape

    def _create_masks(self, seq_len, device):
        mask_f = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask_f = mask_f.masked_fill(mask_f == 0, float("-inf")).masked_fill(mask_f == 1, 0)

        mask_b = torch.triu(torch.ones(seq_len, seq_len, device=device))
        mask_b = mask_b.masked_fill(mask_b == 0, float("-inf")).masked_fill(mask_b == 1, 0)

        mask_d = torch.zeros(seq_len, seq_len, device=device)
        return mask_f, mask_b, mask_d


class TestFusionTransformer:
    def test_initialization(self, small_config):
        model = FusionTransformer(
            vocab_size=small_config.vocab_size,
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            num_layers=small_config.num_layers,
            intermediate_size=small_config.intermediate_size,
        )
        assert model is not None

    def test_forward_diffusion_mode(self, model):
        B, S = 2, 8
        input_ids = torch.randint(1, model.vocab_size, (B, S))
        timesteps = torch.randint(0, 1000, (B,))

        outputs = model(input_ids, timesteps=timesteps, mode=GenerationMode.DIFFUSION)

        assert "hidden" in outputs
        assert "forward_ar_logits" in outputs
        assert "backward_ar_logits" in outputs
        assert "diffusion_features" in outputs

        assert outputs["hidden"].shape == (B, S, model.d_model)
        assert outputs["forward_ar_logits"].shape == (B, S, model.vocab_size)
        assert outputs["backward_ar_logits"].shape == (B, S, model.vocab_size)
        assert outputs["diffusion_features"].shape == (B, S, model.d_model)

    def test_forward_forward_ar_mode(self, model):
        B, S = 2, 8
        input_ids = torch.randint(1, model.vocab_size, (B, S))

        outputs = model(input_ids, mode=GenerationMode.FORWARD_AR)

        assert outputs["hidden"].shape == (B, S, model.d_model)

    def test_forward_backward_ar_mode(self, model):
        B, S = 2, 8
        input_ids = torch.randint(1, model.vocab_size, (B, S))

        outputs = model(input_ids, mode=GenerationMode.BACKWARD_AR)

        assert outputs["hidden"].shape == (B, S, model.d_model)

    def test_forward_with_target_loss(self, model):
        B, S = 2, 8
        input_ids = torch.randint(1, model.vocab_size, (B, S))
        target = torch.randint(0, model.vocab_size, (B, S))

        outputs = model(input_ids, mode=GenerationMode.FORWARD_AR, target=target)

        assert "forward_ar_loss" in outputs
        assert "backward_ar_loss" in outputs
        assert outputs["forward_ar_loss"].item() >= 0
        assert outputs["backward_ar_loss"].item() >= 0

    def test_num_parameters(self, model):
        num_params = model.num_parameters()
        assert num_params > 0

    def test_create_masks(self, model):
        seq_len = 16
        device = torch.device("cpu")
        mask_f, mask_b, mask_d = model.create_masks(seq_len, device)

        assert mask_f.shape == (seq_len, seq_len)
        assert mask_b.shape == (seq_len, seq_len)
        assert mask_d.shape == (seq_len, seq_len)

        assert torch.allclose(mask_f + mask_b, torch.ones(seq_len, seq_len), atol=1e-4)

    def test_num_heads_total_divisible_by_3(self, small_config):
        assert small_config.num_heads_total % 3 == 0


class TestMaskIsolation:
    def test_forward_mask_causal(self, small_config):
        model = FusionTransformer(
            vocab_size=small_config.vocab_size,
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            num_layers=small_config.num_layers,
            intermediate_size=small_config.intermediate_size,
        )

        B, S = 1, 4
        input_ids = torch.randint(1, model.vocab_size, (B, S))
        mask_f, _, _ = model.create_masks(S, input_ids.device)

        for i in range(S):
            for j in range(S):
                if j > i:
                    assert mask_f[i, j] == float("-inf"), f"Forward mask allows future at ({i}, {j})"
                else:
                    assert mask_f[i, j] == 0, f"Forward mask blocks past at ({i}, {j})"

    def test_backward_mask_causal(self, small_config):
        model = FusionTransformer(
            vocab_size=small_config.vocab_size,
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            num_layers=small_config.num_layers,
            intermediate_size=small_config.intermediate_size,
        )

        B, S = 1, 4
        input_ids = torch.randint(1, model.vocab_size, (B, S))
        _, mask_b, _ = model.create_masks(S, input_ids.device)

        for i in range(S):
            for j in range(S):
                if j < i:
                    assert mask_b[i, j] == float("-inf"), f"Backward mask allows past at ({i}, {j})"
                else:
                    assert mask_b[i, j] == 0, f"Backward mask blocks future at ({i}, {j})"

    def test_diffusion_mask_all_visible(self, small_config):
        model = FusionTransformer(
            vocab_size=small_config.vocab_size,
            d_model=small_config.d_model,
            num_heads_total=small_config.num_heads_total,
            num_kv_heads=small_config.num_kv_heads,
            num_layers=small_config.num_layers,
            intermediate_size=small_config.intermediate_size,
        )

        B, S = 1, 4
        input_ids = torch.randint(1, model.vocab_size, (B, S))
        _, _, mask_d = model.create_masks(S, input_ids.device)

        assert torch.allclose(mask_d, torch.zeros(S, S))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

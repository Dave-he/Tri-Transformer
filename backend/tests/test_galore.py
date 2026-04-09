import pytest
import torch
import torch.nn as nn

from app.model.galore_optimizer import GaLoreAdamW, GaLoreProjector


class TestGaLoreProjector:
    def test_project_reduces_rank(self):
        proj = GaLoreProjector(rank=4)
        grad = torch.randn(32, 64)
        low_rank = proj.project(grad)
        assert low_rank.shape[1] == 4

    def test_project_back_shape(self):
        proj = GaLoreProjector(rank=4)
        grad = torch.randn(32, 64)
        low_rank = proj.project(grad)
        restored = proj.project_back(low_rank)
        assert restored.shape == grad.shape

    def test_subspace_updates_after_gap(self):
        proj = GaLoreProjector(rank=4, update_proj_gap=5)
        grad = torch.randn(16, 32)
        proj.project(grad)
        basis_before = proj.ortho_matrix.clone() if proj.ortho_matrix is not None else None
        for _ in range(5):
            proj.project(grad)
        basis_after = proj.ortho_matrix
        assert basis_before is not None
        assert basis_after is not None

    def test_scale_param(self):
        proj = GaLoreProjector(rank=4, scale=0.5)
        grad = torch.randn(16, 32)
        low_rank = proj.project(grad)
        restored = proj.project_back(low_rank)
        assert restored.shape == grad.shape


class TestGaLoreAdamW:
    def _make_model_and_params(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.Linear(64, 32),
        )
        params = []
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                params.append({"params": p, "rank": 4, "update_proj_gap": 10})
            else:
                params.append({"params": p})
        return model, params

    def test_step_updates_params(self):
        model, param_groups = self._make_model_and_params()
        opt = GaLoreAdamW(param_groups, lr=1e-3)
        x = torch.randn(4, 32)
        before = [p.data.clone() for g in param_groups for p in (
            [g["params"]] if isinstance(g["params"], torch.Tensor) else g["params"]
        )]
        loss = model(x).sum()
        loss.backward()
        opt.step()
        after = [p.data for g in param_groups for p in (
            [g["params"]] if isinstance(g["params"], torch.Tensor) else g["params"]
        )]
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed

    def test_zero_grad_clears(self):
        model, param_groups = self._make_model_and_params()
        opt = GaLoreAdamW(param_groups, lr=1e-3)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        for g in param_groups:
            ps = [g["params"]] if isinstance(g["params"], torch.Tensor) else g["params"]
            for p in ps:
                assert p.grad is None or torch.all(p.grad == 0)

    def test_memory_savings_via_state(self):
        model, param_groups = self._make_model_and_params()
        opt = GaLoreAdamW(param_groups, lr=1e-3)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        for g in param_groups:
            ps = [g["params"]] if isinstance(g["params"], torch.Tensor) else g["params"]
            for p in ps:
                if "rank" in g and p.ndim >= 2:
                    state = opt.state[p]
                    assert "exp_avg" in state
                    rank = g["rank"]
                    m_numel = state["exp_avg"].numel()
                    assert m_numel < p.numel()


class TestTrainServiceGaLoreConfig:
    def test_galore_config_validation(self):
        from app.services.train.train_service import validate_galore_config
        cfg = {"use_galore": True, "rank": 128, "update_proj_gap": 200, "scale": 0.25}
        result = validate_galore_config(cfg)
        assert result["use_galore"] is True
        assert result["rank"] == 128

    def test_galore_config_defaults(self):
        from app.services.train.train_service import validate_galore_config
        cfg = {"use_galore": True}
        result = validate_galore_config(cfg)
        assert result["rank"] == 128
        assert result["update_proj_gap"] == 200
        assert result["scale"] == 0.25

    def test_invalid_rank_raises(self):
        from app.services.train.train_service import validate_galore_config
        with pytest.raises(ValueError):
            validate_galore_config({"use_galore": True, "rank": -1})

    def test_no_galore_passthrough(self):
        from app.services.train.train_service import validate_galore_config
        cfg = {"use_galore": False}
        result = validate_galore_config(cfg)
        assert result["use_galore"] is False

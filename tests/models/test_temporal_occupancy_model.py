"""Tests for the stateless CNN-GRU temporal occupancy model."""

import torch
from tensordict import TensorDict

from rsl_rl.models import TemporalOccupancyModel


def _cnn_dims() -> list[dict]:
    return [
        {"type": "reshape", "input_size": 4, "shape": [1, 2, 2]},
        {"type": "conv", "out_channels": 2, "kernel_size": 1},
    ]


def _model(
    distribution_cfg=None, obs_set: str = "actor", output_dim: int = 2
) -> tuple[TemporalOccupancyModel, TensorDict]:
    obs = TensorDict({"policy": torch.randn(3, 5 + 6 * 4)}, batch_size=[3])
    model = TemporalOccupancyModel(
        obs,
        {"actor": ["policy"], "critic": ["policy"]},
        obs_set,
        output_dim,
        hidden_dims=[8],
        temporal_obs_size=24,
        temporal_frames=6,
        frame_size=4,
        cnn_dims=_cnn_dims(),
        gru_hidden_size=8,
        distribution_cfg=distribution_cfg,
        tanh_output=True,
    )
    return model, obs


def test_temporal_occupancy_model_preserves_mlp_head_width_and_forward_shape() -> None:
    model, obs = _model()
    critic, _ = _model(distribution_cfg=None, obs_set="critic", output_dim=1)

    assert model.mlp[0].in_features == 5 + 8
    assert model(obs).shape == (3, 2)
    assert critic(obs).shape == (3, 1)
    assert critic.mlp[0].in_features == 5 + 8
    assert model.get_latent(obs).shape == (3, 5 + 8)


def test_temporal_occupancy_model_uses_final_gru_feature_deterministically() -> None:
    model, obs = _model()
    flat = obs["policy"]
    occupancy = flat[:, -model.temporal_obs_size :]
    frames = occupancy.reshape(-1, model.temporal_frames, model.frame_size)
    cnn_features = model.occupancy_cnn(frames.reshape(-1, model.frame_size))
    expected, _ = model.gru(cnn_features.reshape(-1, model.temporal_frames, model.cnn_output_size))

    latent = model.get_latent(obs)
    assert torch.allclose(latent[:, -model.gru_hidden_size :], expected[:, -1, :])
    assert torch.allclose(model.get_latent(obs), model.get_latent(obs))


def test_temporal_occupancy_model_jit_export_matches_forward() -> None:
    model, obs = _model(
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
    )
    model.eval()
    exported = torch.jit.script(model.as_jit())

    assert torch.allclose(model(obs), exported(obs["policy"]), atol=1.0e-5)

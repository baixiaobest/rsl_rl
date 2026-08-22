"""CNN-GRU model for a temporal occupancy-map tail in a flat observation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.encoder_model import _build_cnn_encoder
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class TemporalOccupancyModel(MLPModel):
    """Encode six occupancy frames with a shared CNN and a per-decision GRU.

    The final ``temporal_obs_size`` entries of the selected flat observations are
    interpreted as chronological ``(frames, frame_size)`` occupancy maps. The
    CNN is applied independently to every frame with shared parameters, then a
    one-layer GRU reduces the sequence to its final hidden feature. The GRU is
    deliberately stateless between policy decisions; temporal context comes
    solely from the six-frame observation tail.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        temporal_obs_size: int = 0,
        temporal_frames: int = 6,
        frame_size: int = 2500,
        cnn_dims: list[dict] | None = None,
        gru_hidden_size: int = 1024,
        tanh_output: bool = False,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        if obs_normalization:
            raise NotImplementedError(
                "TemporalOccupancyModel does not support obs_normalization because its occupancy tail is binary."
            )
        if temporal_frames < 1:
            raise ValueError("temporal_frames must be positive.")
        if temporal_obs_size != temporal_frames * frame_size:
            raise ValueError(
                "temporal_obs_size must equal temporal_frames * frame_size "
                f"({temporal_frames} * {frame_size}), got {temporal_obs_size}."
            )

        _, obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        if temporal_obs_size > obs_dim:
            raise ValueError(
                f"temporal_obs_size ({temporal_obs_size}) exceeds observation dimension ({obs_dim})."
            )
        activation_fn = resolve_nn_activation(activation)

        if cnns is not None:
            if set(cnns.keys()) != {"occupancy"}:
                raise ValueError("Shared temporal occupancy encoders must contain exactly the 'occupancy' module.")
            cnn_dict: nn.ModuleDict | dict[str, nn.Module] = cnns
        else:
            if not cnn_dims:
                raise ValueError("cnn_dims must be provided when the occupancy CNN encoder is not shared.")
            cnn, cnn_input_size, _ = _build_cnn_encoder(cnn_dims, activation_fn)
            if cnn_input_size != frame_size:
                raise ValueError(
                    f"CNN frame input size ({cnn_input_size}) must match frame_size ({frame_size})."
                )
            cnn_dict = {"occupancy": cnn}

        occupancy_cnn = cnn_dict["occupancy"]
        probe_device = next(occupancy_cnn.parameters()).device
        with torch.no_grad():
            cnn_output_size = int(occupancy_cnn(torch.zeros(1, frame_size, device=probe_device)).shape[-1])

        self.temporal_obs_size = temporal_obs_size
        self.temporal_frames = temporal_frames
        self.frame_size = frame_size
        self.cnn_output_size = cnn_output_size
        self.gru_hidden_size = gru_hidden_size
        self.tanh_output = tanh_output
        self.other_obs_size = obs_dim - temporal_obs_size
        self._latent_dim = self.other_obs_size + gru_hidden_size

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        self.cnns = cnn_dict if isinstance(cnn_dict, nn.ModuleDict) else nn.ModuleDict(cnn_dict)
        self.occupancy_cnn = self.cnns["occupancy"]
        self.gru = nn.GRU(cnn_output_size, gru_hidden_size, num_layers=1, batch_first=True)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        mlp_output = self.mlp(self.get_latent(obs, masks, hidden_state))
        if self.tanh_output:
            mlp_output = torch.tanh(mlp_output)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _flat_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def _encode_temporal_occupancy(self, occupancy: torch.Tensor) -> torch.Tensor:
        frames = occupancy.reshape(-1, self.temporal_frames, self.frame_size)
        cnn_features = self.occupancy_cnn(frames.reshape(-1, self.frame_size))
        sequence = cnn_features.reshape(-1, self.temporal_frames, self.cnn_output_size)
        sequence_out, _ = self.gru(sequence)
        return sequence_out[:, -1, :]

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        flat = self._flat_obs(obs)
        other = flat[:, : -self.temporal_obs_size]
        occupancy = flat[:, -self.temporal_obs_size :]
        return torch.cat([other, self._encode_temporal_occupancy(occupancy)], dim=-1)

    def _get_latent_dim(self) -> int:
        return self._latent_dim

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: the model intentionally keeps occupancy values binary."""
        pass

    def as_jit(self) -> nn.Module:
        return _TorchTemporalOccupancyModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxTemporalOccupancyModel(self, verbose)


class _TemporalOccupancyExportBase(nn.Module):
    """Export wrapper operating on the concatenated flat observation vector."""

    def __init__(self, model: TemporalOccupancyModel) -> None:
        super().__init__()
        self.temporal_obs_size = model.temporal_obs_size
        self.temporal_frames = model.temporal_frames
        self.frame_size = model.frame_size
        self.cnn_output_size = model.cnn_output_size
        self.tanh_output = model.tanh_output
        self.occupancy_cnn = copy.deepcopy(model.occupancy_cnn)
        self.gru = copy.deepcopy(model.gru)
        self.mlp = copy.deepcopy(model.mlp)
        self.obs_dim = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        other = x[:, : -self.temporal_obs_size]
        occupancy = x[:, -self.temporal_obs_size :]
        frames = occupancy.reshape(-1, self.temporal_frames, self.frame_size)
        cnn_features = self.occupancy_cnn(frames.reshape(-1, self.frame_size))
        sequence = cnn_features.reshape(-1, self.temporal_frames, self.cnn_output_size)
        sequence_out, _ = self.gru(sequence)
        latent = torch.cat([other, sequence_out[:, -1, :]], dim=-1)
        output = self.mlp(latent)
        if self.tanh_output:
            output = torch.tanh(output)
        return self.deterministic_output(output)


class _TorchTemporalOccupancyModel(_TemporalOccupancyExportBase):
    @torch.jit.export
    def reset(self) -> None:
        """GRU state is local to a forward pass, so export reset is a no-op."""
        pass


class _OnnxTemporalOccupancyModel(_TemporalOccupancyExportBase):
    def __init__(self, model: TemporalOccupancyModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.obs_dim),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

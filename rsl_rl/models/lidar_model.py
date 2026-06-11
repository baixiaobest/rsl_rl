# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation


def _build_cnn(
    layer_configs: list[dict], activation_fn: nn.Module, in_channels: int, height: int, width: int
) -> tuple[nn.Sequential, int]:
    """Build a 2D CNN from a list of layer config dicts.

    Supported layer types: ``conv``, ``pool``, ``adaptive_pool``. Returns ``(nn.Sequential, output_flat_size)``.
    The first ``conv`` layer may omit ``in_channels``; the value inferred from the preceding spatial dimensions is
    used automatically.
    """
    layers: list[nn.Module] = []
    c, h, w = in_channels, height, width

    for i, lc in enumerate(layer_configs):
        ltype = lc.get("type", "conv")

        if ltype == "conv":
            ic = lc.get("in_channels", c)
            oc = lc["out_channels"]
            k = lc.get("kernel_size", 3)
            s = lc.get("stride", 1)
            p = lc.get("padding", 0)
            d = lc.get("dilation", 1)
            conv = nn.Conv2d(ic, oc, k, stride=s, padding=p, dilation=d)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
            layers.append(conv)
            layers.append(activation_fn)
            # update spatial dims
            kh, kw = (k, k) if isinstance(k, int) else k
            ph, pw = (p, p) if isinstance(p, int) else p
            sh, sw = (s, s) if isinstance(s, int) else s
            dh, dw = (d, d) if isinstance(d, int) else d
            h = int((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
            w = int((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)
            c = oc

        elif ltype == "pool":
            k = lc.get("kernel_size", 2)
            s = lc.get("stride", k)
            p = lc.get("padding", 0)
            pt = lc.get("pool_type", "max").lower()
            if pt == "max":
                layers.append(nn.MaxPool2d(k, stride=s, padding=p))
            else:
                layers.append(nn.AvgPool2d(k, stride=s, padding=p))
            kh, kw = (k, k) if isinstance(k, int) else k
            ph, pw = (p, p) if isinstance(p, int) else p
            sh, sw = (s, s) if isinstance(s, int) else s
            h = int((h + 2 * ph - kh) / sh + 1)
            w = int((w + 2 * pw - kw) / sw + 1)

        elif ltype == "adaptive_pool":
            out_size = lc["output_size"]
            layers.append(nn.AdaptiveAvgPool2d(out_size))
            h, w = (out_size, out_size) if isinstance(out_size, int) else out_size

        else:
            raise ValueError(f"Unknown layer type '{ltype}' in lidar_cnn_dims[{i}]")

    layers.append(nn.Flatten())
    return nn.Sequential(*layers), c * h * w


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_fn: nn.Module) -> nn.Sequential:
    """Build a simple MLP with the given activation and a linear output layer."""
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation_fn)
    return nn.Sequential(*layers)


def _build_deconv(
    layer_configs: list[dict], activation_fn: nn.Module, in_channels: int, width: int
) -> tuple[nn.Sequential, int, int]:
    """Build a 1D upsampling (transposed-conv) network from a list of layer config dicts.

    Mirrors :func:`_build_cnn` but with ``nn.ConvTranspose1d`` to *expand* the width back up to the target arc
    resolution. No activation is applied after the final layer (raw regression output). Returns
    ``(nn.Sequential, out_channels, out_width)``.
    """
    layers: list[nn.Module] = []
    c, w = in_channels, width
    n = len(layer_configs)
    for i, lc in enumerate(layer_configs):
        ic = lc.get("in_channels", c)
        oc = lc["out_channels"]
        k = lc.get("kernel_size", 3)
        s = lc.get("stride", 2)
        p = lc.get("padding", 0)
        op = lc.get("output_padding", 0)
        d = lc.get("dilation", 1)
        deconv = nn.ConvTranspose1d(ic, oc, k, stride=s, padding=p, output_padding=op, dilation=d)
        nn.init.kaiming_uniform_(deconv.weight, nonlinearity="relu")
        if deconv.bias is not None:
            nn.init.constant_(deconv.bias, 0)
        layers.append(deconv)
        if i < n - 1:
            layers.append(activation_fn)
        # ConvTranspose1d output length formula
        w = (w - 1) * s - 2 * p + d * (k - 1) + op + 1
        c = oc
    return nn.Sequential(*layers), c, w


class LidarModel(MLPModel):
    """MLP model with a shared 2D-CNN encoder for an embedded temporal-lidar observation.

    The active (1D) observation group is a flat vector whose last ``lidar_obs_size`` entries are a flattened
    ``(C, H, fov_bins)`` lidar tensor and whose remaining entries are proprioceptive ("other") observations. The
    lidar tensor is encoded by a CNN, the other observations by an optional MLP encoder, and the two latents are
    concatenated and passed to the parent MLP head (mapping to actions or a value).

    The CNN encoder is exposed via :attr:`cnns` (a ``ModuleDict`` keyed by ``"lidar"``) and can be shared between the
    actor and critic through the algorithm's ``share_cnn_encoders`` option, exactly like :class:`CNNModel`.

    When ``enable_prediction_head`` is set, an auxiliary next-frame lidar prediction head (an upsampling
    ``ConvTranspose1d`` stack) is attached on top of the shared CNN latent. It is trained by the optional lidar
    prediction phase in PPO via :meth:`predict_next` / :meth:`prediction_parameters`.
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
        # Lidar encoder
        lidar_obs_size: int = 0,
        lidar_horizon: int = 1,
        lidar_fov_bins: int = 1,
        lidar_cnn_dims: list[dict] | None = None,
        other_mlp_dims: list[int] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        # Auxiliary next-frame prediction head
        enable_prediction_head: bool = False,
        pred_cnn_dims: list[dict] | None = None,
        pred_cnn_input_width: int = 8,
        pred_target_channels: int = 1,
    ) -> None:
        """Initialize the lidar model. See class docstring for the observation layout and CNN-sharing behavior."""
        if obs_normalization:
            raise NotImplementedError(
                "LidarModel does not support obs_normalization (the embedded lidar tensor must not be normalized)."
            )

        # Resolve the active (flat) observation group(s) and total dimension using the parent MLP logic.
        _, obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Lidar geometry: infer channel count from the flat lidar size so the model adapts to the observation
        # term's include_validity flag without extra configuration.
        if lidar_obs_size % (lidar_horizon * lidar_fov_bins) != 0:
            raise ValueError(
                f"lidar_obs_size ({lidar_obs_size}) is not divisible by lidar_horizon * lidar_fov_bins "
                f"({lidar_horizon} * {lidar_fov_bins})."
            )
        lidar_channels = lidar_obs_size // (lidar_horizon * lidar_fov_bins)

        activation_fn = resolve_nn_activation(activation)

        # Build (or reuse a shared) lidar CNN encoder. Built as a local variable here; registered as a submodule
        # after the parent constructor runs (nn.Module requires __init__ before assigning submodules).
        if cnns is not None:
            if set(cnns.keys()) != {"lidar"}:
                raise ValueError("Shared CNN encoders for LidarModel must contain exactly the 'lidar' encoder.")
            cnn_dict: nn.ModuleDict | dict[str, nn.Module] = cnns
        else:
            if not lidar_cnn_dims:
                raise ValueError("lidar_cnn_dims must be provided when the CNN encoder is not shared.")
            cnn, _ = _build_cnn(lidar_cnn_dims, activation_fn, lidar_channels, lidar_horizon, lidar_fov_bins)
            cnn_dict = {"lidar": cnn}
        lidar_cnn = cnn_dict["lidar"]

        # Flattened CNN latent size (works for both freshly built and shared encoders).
        with torch.no_grad():
            lidar_latent_dim = int(
                lidar_cnn(torch.zeros(1, lidar_channels, lidar_horizon, lidar_fov_bins)).shape[-1]
            )

        # Build the optional "other observation" encoder.
        other_size = obs_dim - lidar_obs_size
        if other_size < 0:
            raise ValueError(f"lidar_obs_size ({lidar_obs_size}) exceeds the observation dimension ({obs_dim}).")
        if other_mlp_dims:
            other_mlp: nn.Module = _build_mlp(other_size, list(other_mlp_dims[:-1]), other_mlp_dims[-1], activation_fn)
            other_out_dim = other_mlp_dims[-1]
        else:
            other_mlp = nn.Identity()
            other_out_dim = other_size

        # Stash dimensions consumed by the parent constructor (via _get_latent_dim) and by get_latent.
        self.lidar_obs_size = lidar_obs_size
        self.lidar_horizon = lidar_horizon
        self.lidar_fov_bins = lidar_fov_bins
        self.lidar_channels = lidar_channels
        self.lidar_latent_dim = lidar_latent_dim
        self.other_out_dim = other_out_dim
        self._latent_dim = lidar_latent_dim + other_out_dim

        # Initialize the parent MLP model (builds the obs grouping, distribution, and MLP head).
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

        # Register encoders as submodules.
        self.cnns = cnn_dict if isinstance(cnn_dict, nn.ModuleDict) else nn.ModuleDict(cnn_dict)
        self.lidar_cnn = self.cnns["lidar"]
        self.other_mlp = other_mlp

        # Optional next-frame lidar prediction head (world-model auxiliary task). Consumes the lidar CNN latent only
        # (no proprioception) and upsamples it back to a single distance arc of width ``lidar_fov_bins``, so the
        # shared CNN encoder is pressured to learn scene dynamics rather than shortcut via the known ego transform.
        self.enable_prediction_head = enable_prediction_head
        if enable_prediction_head:
            if not pred_cnn_dims:
                raise ValueError("enable_prediction_head=True requires non-empty pred_cnn_dims.")
            pred_in_channels = pred_cnn_dims[0].get("in_channels")
            if pred_in_channels is None:
                raise ValueError("pred_cnn_dims[0] must specify 'in_channels' (initial channel count).")
            self.pred_cnn_input_shape = (pred_in_channels, pred_cnn_input_width)
            self.pred_mlp = _build_mlp(
                lidar_latent_dim, [], pred_in_channels * pred_cnn_input_width, activation_fn
            )
            self.pred_deconv, pred_out_c, pred_out_w = _build_deconv(
                pred_cnn_dims, activation_fn, in_channels=pred_in_channels, width=pred_cnn_input_width
            )
            if pred_out_c != pred_target_channels or pred_out_w != lidar_fov_bins:
                raise ValueError(
                    f"prediction head output ({pred_out_c}ch x {pred_out_w}) does not match target "
                    f"({pred_target_channels}ch x {lidar_fov_bins}). Adjust pred_cnn_dims / pred_cnn_input_width."
                )
            self.pred_target_channels = pred_target_channels
            self.pred_target_size = pred_target_channels * lidar_fov_bins

    # ------------------------------------------------------------------
    # Latent computation
    # ------------------------------------------------------------------

    def _flat_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the active (1D) observation groups into a single flat tensor."""
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)

    def _encode_lidar(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        """Reshape the flat lidar tail into ``(B, C, H, fov_bins)`` and encode it with the shared CNN."""
        x = lidar_flat.view(lidar_flat.shape[0], self.lidar_channels, self.lidar_horizon, self.lidar_fov_bins)
        return self.lidar_cnn(x)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the MLP-head latent by concatenating the CNN lidar latent and the encoded other-obs latent."""
        flat = self._flat_obs(obs)
        other = flat[:, : -self.lidar_obs_size]
        lidar = flat[:, -self.lidar_obs_size :]
        lidar_latent = self._encode_lidar(lidar)
        other_latent = self.other_mlp(other)
        return torch.cat([lidar_latent, other_latent], dim=-1)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self._latent_dim

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: LidarModel does not normalize observations (see __init__ guard)."""
        pass

    # ------------------------------------------------------------------
    # Auxiliary next-frame prediction head
    # ------------------------------------------------------------------

    def predict_next(self, obs: TensorDict) -> torch.Tensor:
        """Predict the next-step lidar distance arc from the lidar CNN latent only.

        Returns ``(B, pred_target_channels * fov_bins)`` to match the stored prediction target layout.
        """
        flat = self._flat_obs(obs)
        lidar = flat[:, -self.lidar_obs_size :]
        lidar_latent = self._encode_lidar(lidar)
        x = self.pred_mlp(lidar_latent)
        x = x.view(x.shape[0], *self.pred_cnn_input_shape)
        x = self.pred_deconv(x)  # (B, C, fov_bins)
        return x.reshape(x.shape[0], -1)

    def prediction_parameters(self):
        """Parameters trained by the auxiliary prediction phase: shared lidar encoder + prediction head."""
        return chain(
            self.lidar_cnn.parameters(),
            self.pred_mlp.parameters(),
            self.pred_deconv.parameters(),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchLidarModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxLidarModel(self, verbose)


class _LidarExportBase(nn.Module):
    """Shared inference logic for exporting a :class:`LidarModel` (operates on the flat observation vector)."""

    def __init__(self, model: LidarModel) -> None:
        super().__init__()
        self.lidar_obs_size = model.lidar_obs_size
        self.lidar_channels = model.lidar_channels
        self.lidar_horizon = model.lidar_horizon
        self.lidar_fov_bins = model.lidar_fov_bins
        self.lidar_cnn = copy.deepcopy(model.lidar_cnn)
        self.other_mlp = copy.deepcopy(model.other_mlp)
        self.mlp = copy.deepcopy(model.mlp)
        self.obs_dim = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on the flat observation vector."""
        other = x[:, : -self.lidar_obs_size]
        lidar = x[:, -self.lidar_obs_size :]
        lidar_2d = lidar.view(lidar.shape[0], self.lidar_channels, self.lidar_horizon, self.lidar_fov_bins)
        latent = torch.cat([self.lidar_cnn(lidar_2d), self.other_mlp(other)], dim=-1)
        return self.deterministic_output(self.mlp(latent))


class _TorchLidarModel(_LidarExportBase):
    """Exportable lidar model for JIT."""

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for lidar exports)."""
        pass


class _OnnxLidarModel(_LidarExportBase):
    """Exportable lidar model for ONNX."""

    def __init__(self, model: LidarModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.obs_dim),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]

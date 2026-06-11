# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class Reshape(nn.Module):
    """Reshape a flat ``(B, N)`` tensor to ``(B, *shape)`` (TorchScript friendly)."""

    def __init__(self, shape: list[int]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.shape) == 3:
            return x.reshape(x.size(0), self.shape[0], self.shape[1], self.shape[2])
        elif len(self.shape) == 2:
            return x.reshape(x.size(0), self.shape[0], self.shape[1])
        elif len(self.shape) == 1:
            return x.reshape(x.size(0), self.shape[0])
        raise ValueError(f"Reshape only supports shapes of length 1, 2, or 3, got {len(self.shape)}.")


def _build_mlp_encoder(encoder_dims: list[int], activation_fn: nn.Module) -> tuple[nn.Sequential, int, int]:
    """Build an MLP encoder ``[input, *hidden, output]`` (no activation after the final layer)."""
    if len(encoder_dims) < 3:
        raise ValueError(
            f"encoder_dims for an MLP encoder must be [input_dim, hidden_dim, ..., output_dim] with at least 3 "
            f"elements, got {encoder_dims}."
        )
    layers: list[nn.Module] = []
    for i in range(len(encoder_dims) - 1):
        layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
        if i < len(encoder_dims) - 2:
            layers.append(activation_fn)
    return nn.Sequential(*layers), encoder_dims[0], encoder_dims[-1]


def _build_cnn_encoder(encoder_dims: list[dict], activation_fn: nn.Module) -> tuple[nn.Sequential, int, int]:
    """Build a CNN encoder from a list of layer config dicts.

    Supported layer types: ``reshape`` (flat -> image; the first layer, must give ``input_size`` and ``shape``),
    ``conv``, ``pool``, ``adaptive_pool``. A trailing ``Flatten`` is appended. Returns
    ``(nn.Sequential, encoder_input_size, encoder_output_size)``.
    """
    if not isinstance(encoder_dims, list) or not all(isinstance(layer, dict) for layer in encoder_dims):
        raise ValueError("For a CNN encoder, encoder_dims must be a list of layer-config dicts.")

    layers: list[nn.Module] = []
    c = h = w = None
    encoder_input_size = None

    for i, lc in enumerate(encoder_dims):
        ltype = lc.get("type", "conv")

        if ltype == "reshape":
            if "shape" not in lc:
                raise ValueError(f"Reshape layer {i} must specify 'shape'.")
            if i == 0:
                encoder_input_size = lc.get("input_size")
                if encoder_input_size is None:
                    raise ValueError("First reshape layer must specify 'input_size'.")
            shape = lc["shape"]
            layers.append(Reshape(shape))
            c, h, w = (shape[0], shape[1], shape[2]) if len(shape) == 3 else (None, None, None)

        elif ltype == "conv":
            if c is None:
                if i == 0 and "in_channels" in lc:
                    c = lc["in_channels"]
                    encoder_input_size = lc.get("input_size", encoder_input_size)
                    if "height" in lc and "width" in lc:
                        h, w = lc["height"], lc["width"]
                    else:
                        raise ValueError("First conv layer must specify 'height' and 'width'.")
                else:
                    raise ValueError(f"Conv layer {i} needs to know its input channels.")
            ic = lc.get("in_channels", c)
            oc = lc.get("out_channels")
            if oc is None:
                raise ValueError(f"Conv layer {i} must specify 'out_channels'.")
            k = lc.get("kernel_size", 3)
            s = lc.get("stride", 1)
            p = lc.get("padding", 0)
            d = lc.get("dilation", 1)
            conv = nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, dilation=d)
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
            layers.append(conv)
            layers.append(activation_fn)
            kh, kw = (k, k) if isinstance(k, int) else k
            ph, pw = (p, p) if isinstance(p, int) else p
            sh, sw = (s, s) if isinstance(s, int) else s
            dh, dw = (d, d) if isinstance(d, int) else d
            c = oc
            h = int((h + 2 * ph - dh * (kh - 1) - 1) / sh + 1)
            w = int((w + 2 * pw - dw * (kw - 1) - 1) / sw + 1)

        elif ltype == "pool":
            k = lc.get("kernel_size", 2)
            s = lc.get("stride", k)
            p = lc.get("padding", 0)
            pt = lc.get("pool_type", "max").lower()
            if pt == "max":
                layers.append(nn.MaxPool2d(kernel_size=k, stride=s, padding=p))
            elif pt == "avg":
                layers.append(nn.AvgPool2d(kernel_size=k, stride=s, padding=p))
            else:
                raise ValueError(f"Unknown pool_type '{pt}'. Should be 'max' or 'avg'.")
            kh, kw = (k, k) if isinstance(k, int) else k
            ph, pw = (p, p) if isinstance(p, int) else p
            sh, sw = (s, s) if isinstance(s, int) else s
            h = int((h + 2 * ph - kh) / sh + 1)
            w = int((w + 2 * pw - kw) / sw + 1)

        elif ltype == "adaptive_pool":
            out_size = lc.get("output_size")
            if out_size is None:
                raise ValueError(f"Adaptive pool layer {i} must specify 'output_size'.")
            layers.append(nn.AdaptiveAvgPool2d(output_size=out_size))
            h, w = (out_size, out_size) if isinstance(out_size, int) else out_size

        else:
            raise ValueError(f"Unknown layer type '{ltype}' in encoder_dims[{i}].")

    layers.append(nn.Flatten())
    if c is None or h is None or w is None:
        raise ValueError("Unable to determine the CNN encoder output size from encoder_dims.")
    return nn.Sequential(*layers), int(encoder_input_size), int(c * h * w)


class EncoderModel(MLPModel):
    """MLP model with an encoder over the tail of a flat observation group.

    The active (1D) observation group is split into ``[main_obs | enc_obs]``, where ``enc_obs`` is the last
    ``encoder_input_size`` entries. ``enc_obs`` is processed by an MLP or CNN encoder, its output is concatenated
    with ``main_obs``, and the result is fed to the parent MLP head. This generalizes the legacy
    ``EncoderActorCritic``: it supports both encoder types, optional per-sample normalization of the encoder input,
    an optional ``tanh`` on the output, and sharing the encoder between actor and critic via the algorithm's
    ``share_cnn_encoders`` option (the encoder is exposed as ``self.cnns["encoder"]``).

    When ``encoder_dims`` is None, no encoder is built and the model is a plain MLP over the full observation (still
    honoring ``tanh_output``), matching the no-encoder configurations of ``EncoderActorCritic``.
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
        encoder_dims: list[int] | list[dict] | None = None,
        encoder_type: str = "mlp",
        encoder_obs_normalize: bool = False,
        tanh_output: bool = False,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the encoder model. See the class docstring for the observation layout and sharing behavior."""
        if obs_normalization:
            raise NotImplementedError(
                "EncoderModel does not support obs_normalization; use encoder_obs_normalize for the encoder input."
            )

        # Resolve the active (flat) observation group(s) and total dimension via the parent MLP logic.
        _, obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        activation_fn = resolve_nn_activation(activation)

        # Build or reuse the (shared) encoder. Built as a local variable; registered after super().__init__.
        if cnns is not None:
            if set(cnns.keys()) != {"encoder"}:
                raise ValueError("Shared encoders for EncoderModel must contain exactly the 'encoder' module.")
            enc_dict: nn.ModuleDict | dict[str, nn.Module] | None = cnns
            encoder = cnns["encoder"]
            encoder_input_size = int(encoder._encoder_input_size)  # type: ignore[attr-defined]
            encoder_output_size = int(encoder._encoder_output_size)  # type: ignore[attr-defined]
        elif encoder_dims is not None:
            if encoder_type == "mlp":
                encoder, encoder_input_size, encoder_output_size = _build_mlp_encoder(encoder_dims, activation_fn)
            elif encoder_type == "cnn":
                encoder, encoder_input_size, encoder_output_size = _build_cnn_encoder(encoder_dims, activation_fn)
            else:
                raise ValueError(f"Unknown encoder_type '{encoder_type}'. Should be 'mlp' or 'cnn'.")
            # Stash dims on the module so a shared (critic) instance can recover them.
            encoder._encoder_input_size = encoder_input_size  # type: ignore[attr-defined]
            encoder._encoder_output_size = encoder_output_size  # type: ignore[attr-defined]
            enc_dict = {"encoder": encoder}
        else:
            encoder = None
            encoder_input_size = 0
            encoder_output_size = 0
            enc_dict = None

        self.encoder_input_size = encoder_input_size
        self.encoder_output_size = encoder_output_size
        self.encoder_obs_normalize = encoder_obs_normalize
        self.tanh_output = tanh_output
        self._latent_dim = obs_dim - encoder_input_size + encoder_output_size

        # Initialize the parent MLP model (builds obs grouping, distribution, and the MLP head over _latent_dim).
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

        # Register the encoder as a submodule (under ``cnns`` to reuse the generic ``share_cnn_encoders`` path).
        if enc_dict is not None:
            self.cnns = enc_dict if isinstance(enc_dict, nn.ModuleDict) else nn.ModuleDict(enc_dict)
            self.encoder: nn.Module | None = self.cnns["encoder"]
        else:
            self.encoder = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass with an optional ``tanh`` applied to the MLP output (the action mean for scalar-std policies)."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.tanh_output:
            mlp_output = torch.tanh(mlp_output)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the MLP-head latent by encoding the observation tail and concatenating it with the head."""
        flat = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        if self.encoder is None:
            return flat
        main = flat[:, : -self.encoder_input_size]
        enc = flat[:, -self.encoder_input_size :]
        if self.encoder_obs_normalize:
            enc = (enc - enc.mean(dim=1, keepdim=True)) / (enc.std(dim=1, keepdim=True) + 1e-8)
        return torch.cat([main, self.encoder(enc)], dim=-1)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self._latent_dim

    def update_normalization(self, obs: TensorDict) -> None:
        """No-op: EncoderModel does not use observation normalization (see __init__ guard)."""
        pass

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxEncoderModel(self, verbose)


class _EncoderExportBase(nn.Module):
    """Shared inference logic for exporting an :class:`EncoderModel` (operates on the flat observation vector)."""

    def __init__(self, model: EncoderModel) -> None:
        super().__init__()
        self.encoder_input_size = model.encoder_input_size
        self.encoder_obs_normalize = model.encoder_obs_normalize
        self.tanh_output = model.tanh_output
        self.encoder = copy.deepcopy(model.encoder) if model.encoder is not None else None
        self.mlp = copy.deepcopy(model.mlp)
        self.obs_dim = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on the flat observation vector."""
        if self.encoder is None:
            latent = x
        else:
            main = x[:, : -self.encoder_input_size]
            enc = x[:, -self.encoder_input_size :]
            if self.encoder_obs_normalize:
                enc = (enc - enc.mean(dim=1, keepdim=True)) / (enc.std(dim=1, keepdim=True) + 1e-8)
            latent = torch.cat([main, self.encoder(enc)], dim=-1)
        out = self.mlp(latent)
        if self.tanh_output:
            out = torch.tanh(out)
        return self.deterministic_output(out)


class _TorchEncoderModel(_EncoderExportBase):
    """Exportable encoder model for JIT."""

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for encoder exports)."""
        pass


class _OnnxEncoderModel(_EncoderExportBase):
    """Exportable encoder model for ONNX."""

    def __init__(self, model: EncoderModel, verbose: bool) -> None:
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

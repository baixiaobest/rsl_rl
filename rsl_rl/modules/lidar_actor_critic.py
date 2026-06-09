from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _build_cnn(layer_configs: list[dict], activation_fn: nn.Module,
               in_channels: int, height: int, width: int):
    """Build a CNN from a list of layer config dicts.

    Supported layer types: ``conv``, ``pool``, ``adaptive_pool``.
    Returns ``(nn.Sequential, output_flat_size)``.

    The first ``conv`` layer may omit ``in_channels``; the value inferred from
    the preceding spatial dimensions is used automatically.
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


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
               activation_fn: nn.Module) -> nn.Sequential:
    """Build a simple MLP with ELU activations and a linear output layer."""
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation_fn)
    return nn.Sequential(*layers)


class LidarActorCritic(nn.Module):
    """Two-stream actor-critic for temporal lidar observations.

    The observation vector is split into two parts:
    - **lidar obs** (last ``lidar_obs_size`` dims): shape ``(B, 2*H*fov_bins)``
      → reshaped to ``(B, 2, H, fov_bins)`` and encoded by a shared CNN.
    - **other obs** (all remaining dims): encoded by a small MLP.

    The latent vectors from both streams are concatenated and fed to separate
    actor and critic MLPs.  The CNN is shared between actor and critic.

    Args:
        num_actor_obs: Total actor observation dimension.
        num_critic_obs: Total critic observation dimension.
        num_actions: Action dimension.
        lidar_obs_size: Size of the flattened lidar observation (``2 * H * fov_bins``).
        lidar_horizon: Number of historical lidar timesteps H.
        lidar_fov_bins: Number of FOV bins.
        lidar_cnn_dims: List of layer config dicts (``type``, ``out_channels``, etc.)
            describing the CNN applied to the ``(2, H, fov_bins)`` lidar input.
        other_mlp_dims: Hidden + output dims for the "other obs" MLP encoder,
            e.g. ``[128, 64]``.  The input dim is inferred automatically.
        actor_hidden_dims: Hidden dims for the actor MLP (maps combined latent → actions).
        critic_hidden_dims: Hidden dims for the critic MLP (maps combined latent → value).
        activation: Activation function name (``"elu"``, ``"relu"``, …).
        init_noise_std: Initial action noise standard deviation.
        noise_std_type: ``"scalar"`` or ``"log"``.
        noise_clip: Tanh-clipping scale for action noise.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        lidar_obs_size: int,
        lidar_horizon: int,
        lidar_fov_bins: int,
        lidar_cnn_dims: list[dict],
        other_mlp_dims: list[int],
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        noise_clip: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "LidarActorCritic.__init__ got unexpected arguments (ignored): "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        activation_fn = resolve_nn_activation(activation)

        self.lidar_obs_size = lidar_obs_size
        self.lidar_horizon = lidar_horizon
        self.lidar_fov_bins = lidar_fov_bins
        self.noise_clip = noise_clip
        self.noise_std_type = noise_std_type

        # Channel count is inferred from the obs size so the model adapts to the
        # observation term's `include_validity` flag (2 channels = [dist, valid],
        # 1 channel = dist only) without any further configuration.
        if lidar_obs_size % (lidar_horizon * lidar_fov_bins) != 0:
            raise ValueError(
                f"lidar_obs_size ({lidar_obs_size}) is not divisible by "
                f"lidar_horizon * lidar_fov_bins ({lidar_horizon} * {lidar_fov_bins})"
            )
        self.lidar_channels = lidar_obs_size // (lidar_horizon * lidar_fov_bins)

        # --- Shared lidar CNN (channels=lidar_channels, height=H, width=fov_bins) ---
        self.lidar_cnn, lidar_latent_dim = _build_cnn(
            lidar_cnn_dims, activation_fn,
            in_channels=self.lidar_channels, height=lidar_horizon, width=lidar_fov_bins,
        )

        # --- Other-obs MLP encoders (actor / critic may differ in size) ---
        actor_other_size = num_actor_obs - lidar_obs_size
        critic_other_size = num_critic_obs - lidar_obs_size

        if other_mlp_dims:
            other_out_dim = other_mlp_dims[-1]
            other_hidden = list(other_mlp_dims[:-1])
            self.actor_other_mlp = _build_mlp(actor_other_size, other_hidden, other_out_dim, activation_fn)
            self.critic_other_mlp = _build_mlp(critic_other_size, other_hidden, other_out_dim, activation_fn)
        else:
            # No MLP encoder — pass raw observations through
            self.actor_other_mlp = nn.Identity()
            self.critic_other_mlp = nn.Identity()
            other_out_dim = actor_other_size  # Identity preserves size

        combined_actor_dim = lidar_latent_dim + other_out_dim
        combined_critic_dim = lidar_latent_dim + (other_out_dim if other_mlp_dims else critic_other_size)

        # --- Actor MLP ---
        self.actor = _build_mlp(combined_actor_dim, list(actor_hidden_dims), num_actions, activation_fn)

        # --- Critic MLP ---
        self.critic = _build_mlp(combined_critic_dim, list(critic_hidden_dims), 1, activation_fn)

        print(f"LidarActorCritic:")
        print(f"  lidar_cnn (output_dim={lidar_latent_dim}): {self.lidar_cnn}")
        print(f"  actor_other_mlp: {self.actor_other_mlp}")
        print(f"  actor: {self.actor}")
        print(f"  critic: {self.critic}")

        # --- Action noise ---
        if noise_std_type == "scalar":
            self.std = nn.Parameter(torch.atanh(init_noise_std / noise_clip * torch.ones(num_actions)))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.atanh(init_noise_std / noise_clip * torch.ones(num_actions))))
        else:
            raise ValueError(f"Unknown noise_std_type: '{noise_std_type}'")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_obs(self, obs: torch.Tensor):
        """Split combined observation into (other_obs, lidar_obs)."""
        other = obs[:, : -self.lidar_obs_size]
        lidar = obs[:, -self.lidar_obs_size:]
        return other, lidar

    def _encode_lidar(self, lidar_flat: torch.Tensor) -> torch.Tensor:
        """Reshape flat lidar vector → (B, 2, H, fov_bins) and pass through CNN."""
        B = lidar_flat.shape[0]
        lidar_2d = lidar_flat.view(B, self.lidar_channels, self.lidar_horizon, self.lidar_fov_bins)
        return self.lidar_cnn(lidar_2d)

    def _get_noise_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        return torch.clamp(torch.tanh(std) * self.noise_clip, min=1e-3)

    # ------------------------------------------------------------------
    # Public interface (matches ActorCritic)
    # ------------------------------------------------------------------

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        other, lidar = self._split_obs(observations)
        lidar_latent = self._encode_lidar(lidar)
        other_latent = self.actor_other_mlp(other)
        mean = self.actor(torch.cat([lidar_latent, other_latent], dim=-1))
        std = self._get_noise_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        other, lidar = self._split_obs(observations)
        lidar_latent = self._encode_lidar(lidar)
        other_latent = self.actor_other_mlp(other)
        return self.actor(torch.cat([lidar_latent, other_latent], dim=-1))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        other, lidar = self._split_obs(critic_observations)
        lidar_latent = self._encode_lidar(lidar)
        other_latent = self.critic_other_mlp(other)
        return self.critic(torch.cat([lidar_latent, other_latent], dim=-1))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

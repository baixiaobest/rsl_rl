from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class EncoderActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        encoder_dims=[12, 64, 64, 16],  # [input_dim, hidden_dim1, hidden_dim2, output_dim]
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        noise_clip: float = 1.0,
        tanh_output: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "EncoderActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Check that encoder_dims has at least 3 elements
        if len(encoder_dims) < 3:
            raise ValueError(
                f"encoder_dims must have at least 3 elements [input_dim, hidden_dim, output_dim], got {encoder_dims}"
            )
        
        self.tanh_output = tanh_output

        activation = resolve_nn_activation(activation)

        # Encoder setup
        encoder_input_size = encoder_dims[0]
        encoder_output_size = encoder_dims[-1]
        
        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(encoder_dims[0], encoder_dims[1]))
        encoder_layers.append(activation)
        for layer_index in range(1, len(encoder_dims) - 1):
            if layer_index == len(encoder_dims) - 2:  # Last layer
                encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
            else:
                encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Modified actor input dimension
        actor_input_size = num_actor_obs - encoder_input_size + encoder_output_size
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_size, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                if self.tanh_output:
                    actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function (same as ActorCritic)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise (same as ActorCritic)
        self.noise_clip = noise_clip
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(torch.atanh(init_noise_std / noise_clip * torch.ones(num_actions)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.atanh(init_noise_std / noise_clip * torch.ones(num_actions))))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Save encoder dimensions for use in forward methods
        self.encoder_input_size = encoder_input_size
        self.encoder_output_size = encoder_output_size

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

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

    def process_observations(self, observations):
        # Split observations
        main_obs = observations[:, :-self.encoder_input_size]
        enc_obs = observations[:, -self.encoder_input_size:]
        
        # Encode the last part
        encoded = self.encoder(enc_obs)
        
        # Concatenate the main observations with the encoded vector
        processed_obs = torch.cat([main_obs, encoded], dim=-1)
        return processed_obs

    def update_distribution(self, observations):
        # Process observations through encoder
        processed_obs = self.process_observations(observations)
        
        # compute mean
        mean = self.actor(processed_obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        std = torch.tanh(std) * self.noise_clip  # Clip the standard deviation
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        # Apply tanh to the actions
        if self.tanh_output:
            actions = torch.tanh(actions)
        return actions

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        processed_obs = self.process_observations(observations)
        actions_mean = self.actor(processed_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the encoder-actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def save_as_jit_script(self, path):
        """
        Save the encoder+actor as a TorchScript scripted module for inference.
        The scripted module will perform the same computation as act_inference().
        """

        class EncoderActorScriptModule(nn.Module):
            def __init__(self, encoder, actor, encoder_input_size):
                super().__init__()
                self.encoder = encoder
                self.actor = actor
                self.encoder_input_size = encoder_input_size

            def forward(self, observations):
                # type: (torch.Tensor) -> torch.Tensor
                main_obs = observations[:, :-self.encoder_input_size]
                enc_obs = observations[:, -self.encoder_input_size:]
                encoded = self.encoder(enc_obs)
                processed_obs = torch.cat([main_obs, encoded], dim=-1)
                actions_mean = self.actor(processed_obs)
                return actions_mean

        script_module = EncoderActorScriptModule(self.encoder, self.actor, self.encoder_input_size)
        scripted = torch.jit.script(script_module)
        scripted.save(path)

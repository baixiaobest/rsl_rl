from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

class Reshape(nn.Module):
    """Helper module for reshaping tensors"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class EncoderActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        encoder_dims=[12, 64, 64, 16],  # [input_dim, hidden_dim1, hidden_dim2, output_dim] for MLP
                                         # or list of dicts for CNN
        encoder_type="mlp",  # "mlp" or "cnn"
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        noise_clip: float = 1.0,
        tanh_output: bool = False,
        **kwargs,
    ):
        """Initialize an encoder-actor-critic model with either MLP or CNN encoders.
        
        The encoder processes a subset of the observation space, and its output is concatenated
        with the remaining observations before being passed to the actor network.
        
        Examples:
            # Example 1: Basic usage with an MLP encoder (default)
            model = EncoderActorCritic(
                num_actor_obs=80,            # Total actor observation dimension
                num_critic_obs=80,           # Total critic observation dimension
                num_actions=12,              # Action dimension
                encoder_dims=[12, 64, 16],   # Last 12 dims of obs processed by encoder, output is 16 dims
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256]
            )
            
            # Example 2: Using a CNN encoder for processing images
            cnn_config = [
                # First layer: reshape flat input to image dimensions
                {
                    'type': 'reshape',
                    'input_size': 768,       # Input size of encoder (768 = 3*16*16)
                    'shape': [3, 16, 16]     # Reshape to 3-channel 16x16 image
                },
                # Convolutional layer
                {
                    'type': 'conv',
                    'out_channels': 32,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                # Max pooling layer
                {
                    'type': 'pool',
                    'kernel_size': 2
                },
                # Another convolutional layer
                {
                    'type': 'conv',
                    'out_channels': 64,
                    'kernel_size': 3,
                    'padding': 1
                }
            ]
            
            model = EncoderActorCritic(
                num_actor_obs=(100 + 768),   # 100 regular obs dims + 768 image dims
                num_critic_obs=(100 + 768),
                num_actions=12,
                encoder_dims=cnn_config,     # CNN configuration
                encoder_type="cnn",          # Specify CNN encoder type
                actor_hidden_dims=[256, 256],
                critic_hidden_dims=[256, 256]
            )
        """
        if kwargs:
            print(
                "EncoderActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.tanh_output = tanh_output
        self.encoder_type = encoder_type

        activation_fn = resolve_nn_activation(activation)

        # Encoder setup
        if encoder_dims is not None:
            if self.encoder_type == "mlp":
                # Check that encoder_dims has at least 3 elements
                if len(encoder_dims) < 3:
                    raise ValueError(
                        f"encoder_dims is either none or has at least 3 elements [input_dim, hidden_dim, output_dim], got {encoder_dims}"
                    )

                encoder_input_size = encoder_dims[0]
                encoder_output_size = encoder_dims[-1]
                
                # MLP Encoder
                encoder_layers = []
                encoder_layers.append(nn.Linear(encoder_dims[0], encoder_dims[1]))
                encoder_layers.append(activation_fn)
                for layer_index in range(1, len(encoder_dims) - 1):
                    if layer_index == len(encoder_dims) - 2:  # Last layer
                        encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
                    else:
                        encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
                        encoder_layers.append(activation_fn)
                self.encoder = nn.Sequential(*encoder_layers)
                
            elif self.encoder_type == "cnn":
                # Build CNN encoder
                self.encoder, encoder_input_size, encoder_output_size = self._build_cnn_encoder(
                    encoder_dims, activation_fn
                )
            
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}. Should be 'mlp' or 'cnn'")
        else:
            self.encoder = None
            encoder_input_size = 0
            encoder_output_size = 0
        
        # Modified actor input dimension
        actor_input_size = num_actor_obs - encoder_input_size + encoder_output_size
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_size, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                if self.tanh_output:
                    actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)

        # Value function (same as ActorCritic)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder ({self.encoder_type}): {self.encoder}")
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
        
    def _build_cnn_encoder(self, encoder_dims, activation_fn):
        """Build a CNN encoder based on the provided configuration.
        
        Args:
            encoder_dims (list): List of dictionaries describing the CNN layers
            activation_fn: Activation function to use after convolutional layers
            
        Returns:
            tuple: (encoder, encoder_input_size, encoder_output_size)
        """
        # For CNN, encoder_dims is a list of dictionaries
        if not isinstance(encoder_dims, list) or not all(isinstance(layer, dict) for layer in encoder_dims):
            raise ValueError("For CNN encoder, encoder_dims must be a list of dictionaries")
            
        encoder_layers = []
        current_channels = None
        current_height = None
        current_width = None
        encoder_input_size = None
        encoder_output_size = None
        
        # Process each layer configuration
        for i, layer_config in enumerate(encoder_dims):
            layer_type = layer_config.get('type', 'conv')
            
            if layer_type == 'reshape':
                # Get the reshape dimensions
                if 'shape' not in layer_config:
                    raise ValueError(f"Reshape layer {i} must specify 'shape'")
                reshape_spec = layer_config['shape']
                
                # Set input size for the first reshape layer
                if i == 0:
                    encoder_input_size = layer_config.get('input_size')
                    if encoder_input_size is None:
                        raise ValueError("First reshape layer must specify 'input_size'")
                
                encoder_layers.append(Reshape(reshape_spec))
                
                # Update current dimensions
                if len(reshape_spec) == 3:  # [channels, height, width]
                    current_channels, current_height, current_width = reshape_spec
                else:
                    current_channels, current_height, current_width = None, None, None
            
            elif layer_type == 'conv':
                # For first layer, need to know input dimensions
                if current_channels is None:
                    if i == 0 and 'in_channels' in layer_config:
                        current_channels = layer_config['in_channels']
                        # Also need input_size, height, width for first conv layer
                        if 'input_size' in layer_config:
                            encoder_input_size = layer_config['input_size']
                        if 'height' in layer_config and 'width' in layer_config:
                            current_height = layer_config['height']
                            current_width = layer_config['width']
                        else:
                            raise ValueError("First conv layer must specify 'height' and 'width'")
                    else:
                        raise ValueError(f"Conv layer {i} needs to know input channels")
                        
                in_channels = layer_config.get('in_channels', current_channels)
                out_channels = layer_config.get('out_channels')
                if out_channels is None:
                    raise ValueError(f"Conv layer {i} must specify 'out_channels'")
                
                kernel_size = layer_config.get('kernel_size', 3)
                stride = layer_config.get('stride', 1)
                padding = layer_config.get('padding', 0)
                dilation = layer_config.get('dilation', 1)
                
                encoder_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation
                ))
                encoder_layers.append(activation_fn)
                
                # Update dimensions
                current_channels = out_channels
                
                # Calculate output dimensions
                if isinstance(kernel_size, tuple):
                    k_h, k_w = kernel_size
                else:
                    k_h, k_w = kernel_size, kernel_size
                
                if isinstance(padding, tuple):
                    p_h, p_w = padding
                else:
                    p_h, p_w = padding, padding
                
                if isinstance(stride, tuple):
                    s_h, s_w = stride
                else:
                    s_h, s_w = stride, stride
                
                if isinstance(dilation, tuple):
                    d_h, d_w = dilation
                else:
                    d_h, d_w = dilation, dilation
                
                current_height = int((current_height + 2 * p_h - d_h * (k_h - 1) - 1) / s_h + 1)
                current_width = int((current_width + 2 * p_w - d_w * (k_w - 1) - 1) / s_w + 1)
                
            elif layer_type == 'pool':
                kernel_size = layer_config.get('kernel_size', 2)
                stride = layer_config.get('stride', kernel_size)
                padding = layer_config.get('padding', 0)
                
                encoder_layers.append(nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ))
                
                # Update dimensions
                if isinstance(kernel_size, tuple):
                    k_h, k_w = kernel_size
                else:
                    k_h, k_w = kernel_size, kernel_size
                
                if isinstance(padding, tuple):
                    p_h, p_w = padding
                else:
                    p_h, p_w = padding, padding
                
                if isinstance(stride, tuple):
                    s_h, s_w = stride
                else:
                    s_h, s_w = stride, stride
                
                current_height = int((current_height + 2 * p_h - k_h) / s_h + 1)
                current_width = int((current_width + 2 * p_w - k_w) / s_w + 1)
        
        # Add a Flatten layer at the end
        encoder_layers.append(nn.Flatten())
        encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the output size (flattened)
        if current_channels is not None and current_height is not None and current_width is not None:
            encoder_output_size = current_channels * current_height * current_width
        else:
            raise ValueError("For CNN encoder, unable to determine final output size")
            
        return encoder, encoder_input_size, encoder_output_size

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
        # Encode the last part
        if not self.encoder is None: 
            # Split observations
            main_obs = observations[:, :-self.encoder_input_size]
            enc_obs = observations[:, -self.encoder_input_size:]
            encoded = self.encoder(enc_obs)
            # Concatenate the main observations with the encoded vector
            processed_obs = torch.cat([main_obs, encoded], dim=-1)
        else:
            # if no encoder is used, just use the main observations
            processed_obs = observations

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
        std = torch.clamp(std, min=1e-6)  # Ensure std is not zero
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
                if self.encoder is None:
                    # If no encoder, just use the main observations
                    processed_obs = main_obs
                else:
                    enc_obs = observations[:, -self.encoder_input_size:]
                    encoded = self.encoder(enc_obs)
                    processed_obs = torch.cat([main_obs, encoded], dim=-1)

                actions_mean = self.actor(processed_obs)
                return actions_mean

        script_module = EncoderActorScriptModule(self.encoder, self.actor, self.encoder_input_size)
        scripted = torch.jit.script(script_module)
        scripted.save(path)

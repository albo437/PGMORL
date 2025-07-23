import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, obj_num=1):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                # Check if we should use flexible CNN
                if base_kwargs.get('use_flexible_cnn', False):
                    if obj_num > 1:
                        base = MOFlexibleCNNBase
                        base_kwargs['obj_num'] = obj_num
                    else:
                        base = FlexibleCNNBase
                else:
                    base = CNNBase
            elif len(obs_shape) == 1:
                if obj_num > 1:
                    base = MOMLPBase
                    base_kwargs['obj_num'] = obj_num
                else:
                    base = MLPBase
            else:
                raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, layernorm=True):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        actor_modules = []
        last_hidden_size = num_inputs
        for _ in range(2):
            actor_modules.append(init_(nn.Linear(last_hidden_size, hidden_size, bias=not layernorm)))
            if layernorm:
                actor_modules.append(nn.LayerNorm(hidden_size, elementwise_affine=True))
            actor_modules.append(nn.Tanh())
            last_hidden_size = hidden_size

        self.actor = nn.Sequential(*actor_modules)

        critic_modules = []
        last_hidden_size = num_inputs
        for _ in range(2):
            critic_modules.append(init_(nn.Linear(last_hidden_size, hidden_size, bias=not layernorm)))
            if layernorm:
                critic_modules.append(nn.LayerNorm(hidden_size, elementwise_affine=True))
            critic_modules.append(nn.Tanh())
            last_hidden_size = hidden_size

        self.critic = nn.Sequential(*critic_modules)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MOMLPBase(MLPBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, layernorm=True, obj_num=2):
        super().__init__(num_inputs, recurrent, hidden_size, layernorm)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.critic_linear = init_(nn.Linear(hidden_size, obj_num))
        
        self.train()


class FlexibleCNNBase(NNBase):
    """
    Flexible CNN base class that can be configured with arbitrary CNN architectures.
    Supports multi-objective outputs and configurable layers.
    """
    
    def __init__(self, num_inputs, recurrent=False, cnn_layers=None, 
                 cnn_hidden_size=512, cnn_final_layers=None, cnn_activation='relu',
                 obj_num=1, layernorm=False):
        super(FlexibleCNNBase, self).__init__(recurrent, cnn_hidden_size, cnn_hidden_size)
        
        self.obj_num = obj_num
        
        # Default CNN layers if none provided (Nature CNN style)
        if cnn_layers is None:
            cnn_layers = [[32, 8, 4, 0], [64, 4, 2, 0], [64, 3, 1, 0]]
        
        if cnn_final_layers is None:
            cnn_final_layers = [cnn_hidden_size]
        
        # Determine final hidden size
        self._hidden_size = cnn_final_layers[-1]
        
        # Initialize conv layers
        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), nn.init.calculate_gain('relu'))
        
        # Build convolutional layers
        conv_layers = []
        in_channels = num_inputs
        
        for i, layer_config in enumerate(cnn_layers):
            if len(layer_config) == 3:
                out_channels, kernel_size, stride = layer_config
                padding = 0
            else:
                out_channels, kernel_size, stride, padding = layer_config
            
            # Handle different kernel_size formats
            if isinstance(kernel_size, list):
                kernel_size = tuple(kernel_size)
            
            conv_layers.extend([
                init_relu(nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding)),
                nn.ReLU() if cnn_activation == 'relu' else nn.Tanh()
            ])
            in_channels = out_channels
        
        conv_layers.append(Flatten())
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate the size of flattened features dynamically
        self.conv_output_size = self._calculate_conv_output_size(num_inputs)
        
        # Build fully connected layers
        fc_layers = []
        fc_sizes = [self.conv_output_size] + cnn_final_layers
        
        init_fc = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                constant_(x, 0), np.sqrt(2))
        
        for i in range(len(fc_sizes) - 1):
            fc_layers.append(init_fc(nn.Linear(fc_sizes[i], fc_sizes[i + 1])))
            
            if layernorm:
                fc_layers.append(nn.LayerNorm(fc_sizes[i + 1]))
                
            fc_layers.append(nn.ReLU() if cnn_activation == 'relu' else nn.Tanh())
        
        self.fc_net = nn.Sequential(*fc_layers)
        
        # GRU for recurrent processing (if needed)
        if recurrent:
            self.gru = nn.GRU(self._hidden_size, cnn_hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
        
        # Output layers
        init_output = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                    constant_(x, 0))
        
        # Multi-objective critic (for MORL)
        if obj_num > 1:
            self.critic_linear = init_output(nn.Linear(self._hidden_size, obj_num))
        else:
            self.critic_linear = init_output(nn.Linear(self._hidden_size, 1))
        
        self.train()
    
    def _calculate_conv_output_size(self, num_inputs, input_size=(84, 84)):
        """Calculate the output size after all conv layers with a dummy forward pass."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_inputs, *input_size)
            try:
                conv_output = self.conv_net(dummy_input)
                return conv_output.shape[1]
            except RuntimeError:
                # If 84x84 doesn't work, try common sizes
                for size in [(96, 96), (64, 64), (128, 128), (48, 48)]:
                    try:
                        dummy_input = torch.zeros(1, num_inputs, *size)
                        conv_output = self.conv_net(dummy_input)
                        return conv_output.shape[1]
                    except RuntimeError:
                        continue
                # If all fail, return a reasonable default
                return 1024
    
    @property
    def output_size(self):
        return self._hidden_size
    
    def forward(self, inputs, rnn_hxs, masks):
        # Extract features with CNN
        x = self.conv_net(inputs / 255.0)  # Normalize image inputs
        
        # Process through fully connected layers
        x = self.fc_net(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MOFlexibleCNNBase(FlexibleCNNBase):
    """Multi-objective version of FlexibleCNNBase."""
    
    def __init__(self, num_inputs, obj_num=2, **kwargs):
        kwargs['obj_num'] = obj_num
        super(MOFlexibleCNNBase, self).__init__(num_inputs, **kwargs)
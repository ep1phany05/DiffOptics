import numpy as np
import torch
from torch import nn

from inr import SineLayer


class DINER(nn.Module):
    def __init__(
        self,
        hash_mod: bool = True,  # Whether to use hash table modulation.
        hash_table_length: int = 512 * 512,  # Length of the hash table.
        in_features: int = 3,  # Number of input features.
        hidden_features: int = 64,  # Number of hidden units in each layer.
        hidden_layers: int = 2,  # Number of hidden layers.
        out_features: int = 3,  # Number of output features.
        outermost_linear: bool = True,  # Whether the final layer is linear or a sine layer.
        first_omega_0: float = 6.0,  # Frequency parameter for the first sine layer.
        hidden_omega_0: float = 6.0  # Frequency parameter for the hidden sine layers.
    ):
        super(DINER, self).__init__()
        self.hash_mod = hash_mod
        self.table = nn.Parameter(1e-4 * (torch.rand((hash_table_length, in_features)) * 2 - 1), requires_grad=True)
        
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        layers += [SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0) for _ in range(hidden_layers)]
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
        else:
            output = self.net(coords)
        output = torch.clamp(output, min=-1.0, max=1.0)
        return {"model_out": output, "table": self.table}
    
    def load_pretrained(self, model_path, device=None):
        if device:
            self.to(device)
            checkpoint = torch.load(model_path, map_location=device)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint["net"])


class DINER_XY_polynomial(nn.Module):
    def __init__(
        self,
        hash_mod: bool = True,  # Whether to use hash table modulation.
        hash_table_length: int = 512 * 512,  # Length of the hash table.
        out_mask: list = None,  # Mask for the output features.
        in_features: int = 3,  # Number of input features.
        hidden_features: int = 64,  # Number of hidden units in each layer.
        hidden_layers: int = 2,  # Number of hidden layers.
        out_features: int = 3,  # Number of output features.
        outermost_linear: bool = True,  # Whether the final layer is linear or a sine layer.
        first_omega_0: float = 6.0,  # Frequency parameter for the first sine layer.
        hidden_omega_0: float = 6.0  # Frequency parameter for the hidden sine layers.
    ):
        super(DINER_XY_polynomial, self).__init__()
        
        # Mask for the output features.
        if out_mask is None:
            out_mask = [1] * 10
        self.out_mask = out_mask
        self.hash_table_length = sum(out_mask)
        print(f"Effective hash_table_length: {self.hash_table_length}")
        
        self.hash_mod = hash_mod
        self.table = nn.Parameter(1e-4 * (torch.rand((self.hash_table_length, in_features)) * 2 - 1), requires_grad=True)
        
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        layers += [SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0) for _ in range(hidden_layers)]
        
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords):
        if self.hash_mod:
            partial_output = self.net(self.table)
        else:
            partial_output = self.net(coords)
        
        # 对 partial_output 做 clamping 等处理
        partial_output = torch.clamp(partial_output, min=-1.0, max=1.0) * 1e-10
        full_output = torch.zeros(len(self.out_mask), partial_output.shape[-1], device=partial_output.device)
        idx_trainable = 0
        for i, m in enumerate(self.out_mask):
            if m == 1:
                full_output[i] = partial_output[idx_trainable]
                idx_trainable += 1
            else:
                pass
        
        return {"model_out": full_output, "table": self.table}
    
    def load_pretrained(self, model_path, device=None):
        if device:
            self.to(device)
            checkpoint = torch.load(model_path, map_location=device)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint["net"])

# Ported from StarVLA @ 985c223 (NVIDIA GR00T code, Apache-2.0):
#   swish, SinusoidalPositionalEncoding <- starVLA/model/modules/action_model/flow_matching_head/action_encoder.py:20-54
#   MLP, ActionEncoder                 <- starVLA/model/modules/action_model/GR00T_ActionHeader.py:52-101
# The PI head's copies (LayerwiseFM_ActionHeader.py:51-100) have identical bodies and differ only in
# default arguments (MLP hidden_dim=1024, output_dim=2048; ActionEncoder hidden_size=1024). One copy is
# kept here WITHOUT defaults; the PI head passes the fork's defaults explicitly (change C3 in __init__.py).
# Bodies are verbatim (comments shortened).

import torch
import torch.nn as nn
import torch.nn.functional as F


# fork: flow_matching_head/action_encoder.py:20-21
def swish(x):
    return x * torch.sigmoid(x)


# fork: flow_matching_head/action_encoder.py:24-54
class SinusoidalPositionalEncoding(nn.Module):
    """Produces a sinusoidal encoding of shape (B, T, w) given timesteps of shape (B, T)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


# fork: GR00T_ActionHeader.py:52-59 (= LayerwiseFM_ActionHeader.py:51-58 up to defaults)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


# fork: GR00T_ActionHeader.py:62-101 (= LayerwiseFM_ActionHeader.py:61-100 up to defaults)
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """actions: (B, T, action_dim); timesteps: (B,) one scalar per batch item -> (B, T, hidden_size)"""
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps => (B, T)
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x

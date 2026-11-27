"""Controlled structural variants of R2-D for the NSNet3 structure-transfer round.

Namespace: ``ns3st``.  Nothing here replaces or imports-over an existing
candidate; S0 is a byte-compatible copy of R2-D's module tree so that the
frozen C0 checkpoint loads with ``strict=True``.

Frozen reference (S0): two ``BandMixBlock`` blocks, dense 33x33 per-channel
band mixing + one causal full-channel time GRU (H24) + PReLU residual.

Single-factor variants (each changes exactly one thing relative to S0):

* ``S-T``  grouped time recurrence: the H24 full-channel time GRU becomes two
  independent groups (input 8 -> hidden 8) concatenated to 16, one state per
  group, kept per band; output projection 16->16 kept, residual kept.
* ``S-A``  one GTCRN-style temporal recurrent attention (energy over frequency
  -> GRU(16->8) -> Linear(8->16) -> sigmoid) inserted after the time output
  projection and before the residual, on both blocks.  Gate is
  ``g = 2*sigmoid(.)`` with a zero-initialised output linear, so g == 1 at
  step 0 (this differs from the original GT sigmoid gate and is registered as
  such).
* ``S-M``  dense 33x33 per-channel mixing replaced by a rank-8 33->8->33
  low-rank map; everything else unchanged.

These are R2-D adaptations inspired by traceable GTCRN/NSNet3 modules; they
are not claimed to be the same modules as NSNet3's.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

CH = 16
BANDS = 33
from _backbone_base import Cap2Net, N_BANDS

CANDIDATES = ("S0", "S-T", "S-A", "S-M")


class DenseBandMix(nn.Module):
    """S0: per-channel dense band mixing (same parameter name as R2-D)."""

    def __init__(self, ch, bands=BANDS):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ch, bands, bands))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):                        # [B,C,T,F]
        z = x.permute(0, 1, 3, 2)                # [B,C,F,T]
        z = torch.matmul(self.weight, z)         # [C,F,F] x [B,C,F,T]
        return z.permute(0, 1, 3, 2)


class LowRankBandMix(nn.Module):
    """S-M: rank-8 33 -> 8 -> 33 per-channel map."""

    def __init__(self, ch, bands=BANDS, rank=8):
        super().__init__()
        self.rank = rank
        self.w1 = nn.Parameter(torch.empty(ch, rank, bands))   # 33 -> 8
        self.w2 = nn.Parameter(torch.empty(ch, bands, rank))   # 8 -> 33
        # init is overwritten from the parent SVD in make_init.py
        nn.init.zeros_(self.w1)
        nn.init.zeros_(self.w2)

    def forward(self, x):                        # [B,C,T,F]
        z = x.permute(0, 1, 3, 2)                # [B,C,F,T]
        z = torch.matmul(self.w1, z)             # [B,C,8,T]
        z = torch.matmul(self.w2, z)             # [B,C,F,T]
        return z.permute(0, 1, 3, 2)


class TimeRecurrentAttention(nn.Module):
    """S-A: frequency-energy recurrent gate (R2-D-adapted TRA variant).

    energy_t = mean_F z^2            -> [B,C,T]
    gate     = 2 * sigmoid(Linear(GRU(energy)))
    The output linear is zero-initialised, so gate == 1 at step 0.
    """

    def __init__(self, ch=CH, hidden=8):
        super().__init__()
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, ch)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z, state=None):            # z [B,C,T,F]
        e = z.pow(2).mean(dim=-1)                # [B,C,T]
        a, state = self.gru(e.transpose(1, 2), state)
        a = self.fc(a).transpose(1, 2)           # [B,C,T]
        return z * (2.0 * torch.sigmoid(a)).unsqueeze(-1), state

    def init_state(self, batch, bands, device=None):
        # one stream per band, hidden 8
        return torch.zeros(1, batch * bands, 8, device=device)


class BandMixBlock(nn.Module):
    """One bottleneck block; ``variant`` selects the single-factor change."""

    def __init__(self, ch, hidden, variant="S0", groups=2, rank=8,
                 group_hidden=8, tra_hidden=8):
        super().__init__()
        self.variant = variant
        self.hidden = hidden
        if variant == "S-M":
            self.mix = LowRankBandMix(ch, BANDS, rank)
        else:
            self.mix = DenseBandMix(ch, BANDS)
        if variant == "S-T":
            self.groups = groups
            self.group_ch = ch // groups
            self.group_hidden = group_hidden
            # two independent GRUs, one per channel group (GTCRN GRNN style)
            self.time = nn.ModuleList(
                [nn.GRU(self.group_ch, self.group_hidden, batch_first=True)
                 for _ in range(groups)])
            self.time_fc = nn.Linear(groups * self.group_hidden, ch)
        elif variant == "S-A":
            self.time = nn.GRU(ch, hidden, batch_first=True)
            self.time_fc = nn.Linear(hidden, ch)
            self.tra = TimeRecurrentAttention(ch, tra_hidden)
        else:
            self.time = nn.GRU(ch, hidden, batch_first=True)
            self.time_fc = nn.Linear(hidden, ch)
        self.act = nn.PReLU()

    # -- state helpers ---------------------------------------------------
    def state_len(self, batch=1, bands=BANDS):
        """Flat element count in this block's exported state tensor."""
        if self.variant == "S-T":
            return self.groups * batch * bands * self.group_hidden
        if self.variant == "S-A":
            return batch * bands * self.hidden + batch * self.tra.gru.hidden_size
        return batch * bands * self.hidden

    def init_state(self, batch, bands=BANDS, device=None):
        return torch.zeros(1, self.state_len(batch, bands), device=device)

    def forward(self, x, state=None):            # [B,C,T,F]
        b, c, t, f = x.shape
        x = self.act(x + self.mix(x))
        z = x.permute(0, 3, 2, 1).reshape(b * f, t, c)      # [B*F,T,C]
        if self.variant == "S-T":
            g, cg = self.groups, self.group_ch
            n = b * f
            if state is None:
                s = None
            else:
                s = state.reshape(1, g * n, self.group_hidden)
            outs, new_s = [], []
            for k in range(g):
                sk = None if s is None else s[:, k * n:(k + 1) * n]
                zk, sk = self.time[k](z[:, :, k * cg:(k + 1) * cg], sk)
                outs.append(zk)
                new_s.append(sk)
            z = torch.cat(outs, dim=-1)                       # [B*F,T,C]
            z = self.time_fc(z)
            state = torch.cat([x.reshape(1, -1) for x in new_s], dim=-1)
        elif self.variant == "S-A":
            h = self.hidden
            n_t = b * f * h
            if state is None:
                tstate = astate = None
            else:
                tstate = state[..., :n_t].reshape(1, b * f, h)
                astate = state[..., n_t:].reshape(1, b, self.tra.gru.hidden_size)
            z, tstate = self.time(z, tstate)
            z = self.time_fc(z)
            z = z.reshape(b, f, t, c).permute(0, 3, 2, 1)
            z, astate = self.tra(z, astate)
            state = torch.cat([tstate.reshape(1, -1), astate.reshape(1, -1)], dim=-1)
            return self.act(x + z), state
        else:
            if state is not None:
                state = state.reshape(1, b * f, self.hidden)
            z, state = self.time(z, state)
            z = self.time_fc(z)
            state = state.reshape(1, -1)
        z = z.reshape(b, f, t, c).permute(0, 3, 2, 1)
        return self.act(x + z), state


class NS3STNet(Cap2Net):
    """Cap2Net with an ns3st bottleneck variant; S0 is module-tree R2-D."""

    def __init__(self, candidate: str):
        if candidate not in CANDIDATES:
            raise ValueError(candidate)
        super().__init__(backbone="S", head="CRM", ch=CH, inter=24,
                         hidden=40, s_hidden=38, blocks=2)
        self.candidate = candidate
        self.bottleneck = nn.ModuleList(
            [BandMixBlock(CH, 24, candidate), BandMixBlock(CH, 24, candidate)])
        self.kind = "ns3st"

    def init_state(self, batch, device=None):
        return [blk.init_state(batch, BANDS, device) for blk in self.bottleneck]

    def state_shapes(self, batch: int = 1):
        return [blk.init_state(batch, BANDS).numel() for blk in self.bottleneck]

    def signature(self):
        return {"family": "ns3st", "candidate": self.candidate,
                "n_fft": 512, "hop": 256, "ch": CH, "head": "CRM"}


def state_shapes(candidate: str, batch: int = 1):
    net = NS3STNet(candidate)
    return net.state_shapes(batch)
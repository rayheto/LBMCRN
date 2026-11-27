"""Capacity-v2 architectures: G/S backbones with IRM/CRM/DF heads.

Frozen decisions (EXECUTION_PLAN §9, PROTOCOL_FROZEN):

* native 16 kHz, 512-sample (32 ms) causal analysis window, 256-sample
  (16 ms) hop, every hop updates;
* fixed 257 -> 129 band map: 65 low bins kept, 192 high bins mapped to 64
  triangular ERB bands (GTCRN's invariant mapping); the same matrix expands
  back to 257 bins;
* features are [log1p|X|, cos(phase), sin(phase)] so no running statistic is
  needed and the transform is causal;
* G = global bottleneck (project -> GRU -> project, residual);
  S = shared-subband backbone (frequency bidirectional GRU + causal time GRU,
  weights shared across bands, explicit per-band recurrent state);
* heads: IRM (bounded real mask), CRM (complex mask, range fixed), DF-K
  (band gain times an identity-plus-residual K-tap complex filter).

Only the final head layer is zero-initialised; earlier layers keep their
standard initialisation so gradients start flowing after the first update.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

NFFT = 512
HOP = 256
NBINS = NFFT // 2 + 1          # 257
N_LOW = 65
N_BANDS = N_LOW + 64           # 129
MACROS = None


def _erb_filters(n_fft=512, n_low=65, n_high=64, fs=16000, high_lim=8000):
    """GTCRN's fixed low-bin split plus triangular ERB bands above n_low."""
    low_lim = n_low / n_fft * fs

    def hz2erb(hz):
        return 21.4 * np.log10(0.00437 * hz + 1)

    def erb2hz(e):
        return (10 ** (e / 21.4) - 1) / 0.00437

    points = np.linspace(hz2erb(low_lim), hz2erb(high_lim), n_high)
    bins = np.round(erb2hz(points) / fs * n_fft).astype(np.int32)
    f = np.zeros([n_high, n_fft // 2 + 1], dtype=np.float64)
    f[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
        / (bins[1] - bins[0] + 1e-12)
    for i in range(n_high - 2):
        f[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
            / (bins[i + 1] - bins[i] + 1e-12)
        f[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
            / (bins[i + 2] - bins[i + 1] + 1e-12)
    f[-1, bins[-2]:bins[-1] + 1] = 1 - f[-2, bins[-2]:bins[-1] + 1]
    f = np.abs(f[:, n_low:])
    return torch.from_numpy(f).float()          # [64, 192]


class BandMap(nn.Module):
    """Fixed 257 -> 129 map and its 129 -> 257 expansion."""

    def __init__(self):
        super().__init__()
        hi = _erb_filters()
        self.register_buffer("hi", hi)                       # [64,192]
        self.register_buffer("hi_t", hi.t().contiguous())    # [192,64]

    def bm(self, x):
        """x [..., 257] -> [..., 129]."""
        return torch.cat([x[..., :N_LOW], x[..., N_LOW:] @ self.hi.t()], dim=-1)

    def bs(self, x):
        """x [..., 129] -> [..., 257]."""
        return torch.cat([x[..., :N_LOW], x[..., N_LOW:] @ self.hi_t.t()], dim=-1)


class CausalDsp(nn.Module):
    """Causal 512/256 STFT and OLA with sqrt-Hann analysis/synthesis."""

    def __init__(self, n_fft=NFFT, hop=HOP):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.window = torch.hann_window(n_fft).pow(0.5)

    def analysis(self, x):
        # Right-pad to a whole hop so the flush frame exists; zeros only.
        t = x.shape[-1]
        n = int(math.ceil(t / self.hop)) * self.hop
        pad = F.pad(x, (self.n_fft, n - t))
        spec = torch.stft(pad, self.n_fft, self.hop, self.n_fft,
                          self.window.to(x.device), return_complex=True,
                          center=False)
        return spec.permute(0, 2, 1)                 # [B,T,F]

    def render(self, spec, length=None):
        """Explicit weighted overlap-add.

        ``torch.istft(center=False)`` rejects the zero window-sum at the first
        and last half window.  Those samples lie in the burn-in/padding region,
        so a masked division is exact everywhere that matters.
        """
        b, t, f = spec.shape
        frames = torch.fft.irfft(spec, n=self.n_fft, dim=-1)      # [B,T,nfft]
        win = self.window.to(spec.device)
        frames = frames * win
        pad_len = (t - 1) * self.hop + self.n_fft
        y = F.fold(frames.transpose(1, 2), output_size=(1, pad_len),
                   kernel_size=(1, self.n_fft), stride=(1, self.hop))[:, 0, 0]
        w = F.fold((win * win).view(1, -1, 1).expand(b, -1, t),
                   output_size=(1, pad_len), kernel_size=(1, self.n_fft),
                   stride=(1, self.hop))[:, 0, 0]
        y = y / w.clamp(min=1e-8)
        y = y[..., self.n_fft:]
        if length is not None:
            y = y[..., :length]
        return y


def features_from_spec(spec):
    """Complex [B,T,257] -> normalized [B,3,T,129] (all causal)."""
    mag = spec.abs()
    logmag = torch.log1p(mag)
    phase = spec / mag.clamp(min=1e-6)
    feat = torch.stack([logmag, phase.real, phase.imag], dim=1)
    return feat


class SFE(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, x):                       # [B,C,T,F]
        p = (self.k - 1) // 2
        padded = torch.cat([x[..., :p], x, x[..., -p:]], dim=-1)
        taps = [padded[..., i:i + x.shape[3]] for i in range(self.k)]
        taps = torch.stack(taps, dim=2)          # [B,C,k,T,F]
        return taps.reshape(x.shape[0], x.shape[1] * self.k, x.shape[2], x.shape[3])


class CausalConv2d(nn.Module):
    """Conv2d with left-only time padding (no future frames)."""

    def __init__(self, in_ch, out_ch, kernel, stride=(1, 1), dilation=(1, 1),
                 groups=1, bias=True, pad_freq=None):
        super().__init__()
        self.pad_t = (kernel[0] - 1) * dilation[0]
        self.pad_f = (kernel[1] - 1) * dilation[1] // 2 if pad_freq is None else pad_freq
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Build zero padding with Concat instead of Pad: the ESP-DL Pad
        # exporter drops the string mode and aborts on load.
        b, c, t, f = x.shape
        parts = []
        if self.pad_t:
            parts.append(x.new_zeros(b, c, self.pad_t, f))
        parts.append(x)
        if self.pad_t:
            parts.append(x.new_zeros(b, c, self.pad_t, f))
        # only the left time pad is kept; drop the right one again
        x = torch.cat(parts, dim=2)[:, :, : t + self.pad_t]
        if self.pad_f:
            left = x.new_zeros(b, c, x.shape[2], self.pad_f)
            right = x.new_zeros(b, c, x.shape[2], self.pad_f)
            x = torch.cat([left, x, right], dim=3)
        return self.conv(x)


class SepTF(nn.Module):
    """Causal separable time-frequency block with a residual."""

    def __init__(self, ch, dilation_t=1):
        super().__init__()
        self.dw = CausalConv2d(ch, ch, (3, 3), dilation=(dilation_t, 1),
                               groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.act = nn.PReLU(ch)

    def forward(self, x):
        return self.act(x + self.pw(self.dw(x)))


class GBlock(nn.Module):
    def __init__(self, ch, bands, inter, hidden):
        super().__init__()
        d = ch * bands
        self.fc_in = nn.Linear(d, inter)
        self.gru = nn.GRU(inter, hidden, batch_first=True)
        self.fc_out = nn.Linear(hidden, d)
        self.act = nn.PReLU()

    def forward(self, x, state=None):          # [B,C,T,F]
        b, c, t, f = x.shape
        h = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        h = self.act(self.fc_in(h))
        h, state = self.gru(h, state)
        h = self.fc_out(h).reshape(b, t, c, f).permute(0, 2, 1, 3)
        return self.act(x + h), state


class SBlock(nn.Module):
    """Shared-subband block: freq bi-GRU + causal time GRU, both shared."""

    def __init__(self, ch, hidden):
        super().__init__()
        self.freq = nn.GRU(ch, hidden, batch_first=True, bidirectional=True)
        self.freq_fc = nn.Linear(2 * hidden, ch)
        self.time = nn.GRU(ch, hidden, batch_first=True)
        self.time_fc = nn.Linear(hidden, ch)
        self.act = nn.PReLU()

    def forward(self, x, state=None):          # [B,C,T,F]
        b, c, t, f = x.shape
        # frequency path, one independent sequence per frame
        z = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        z, _ = self.freq(z)
        z = self.freq_fc(z).reshape(b, t, f, c).permute(0, 3, 1, 2)
        x = self.act(x + z)
        # causal time path, one sequence per band
        z = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        z, state = self.time(z, state)
        z = self.time_fc(z).reshape(b, f, t, c).permute(0, 3, 2, 1)
        return self.act(x + z), state


class GroupedBandMix(nn.Module):
    """Block-diagonal linear mixing across bands (shared over time)."""

    def __init__(self, ch, bands=33, groups=4):
        super().__init__()
        self.groups = groups
        base = bands // groups
        sizes = [base] * groups
        sizes[-1] += bands - base * groups
        self.sizes = sizes
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(ch, n, n)) for n in sizes])
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x):                        # [B,C,T,F]
        out = []
        pos = 0
        for w, n in zip(self.weights, self.sizes):
            z = x[..., pos:pos + n]
            out.append(torch.einsum("bctf,cfg->bctg", z, w))
            pos += n
        return torch.cat(out, dim=-1)


class SBlockG(nn.Module):
    """Grouped frequency mixing + shared causal time GRU (+ optional dense mix)."""

    def __init__(self, ch, hidden, groups=4, dense_mix=False, bands=33,
                 gate=False):
        super().__init__()
        self.band = GroupedBandMix(ch, bands, groups)
        self.dense = nn.Linear(bands, bands, bias=False) if dense_mix else None
        self.gate = nn.Conv2d(ch, ch, 1) if gate else None
        self.time = nn.GRU(ch, hidden, batch_first=True)
        self.time_fc = nn.Linear(hidden, ch)
        self.act = nn.PReLU()

    def forward(self, x, state=None):
        b, c, t, f = x.shape
        x = self.act(x + self.band(x))
        if self.dense is not None:
            # x is [B,C,T,F]; the dense mix acts on the band axis.
            x = self.act(x + self.dense(x))
        if self.gate is not None:
            x = x * torch.sigmoid(self.gate(x))
        z = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        z, state = self.time(z, state)
        z = self.time_fc(z).reshape(b, f, t, c).permute(0, 3, 2, 1)
        return self.act(x + z), state


class Cap2Net(nn.Module):
    def __init__(self, backbone="G", head="IRM", ch=16, inter=32, hidden=32,
                 s_hidden=40, blocks=2, df_order=3, s_mode="rnn",
                 s_groups=4, s_mix=False, s_gate=False):
        super().__init__()
        if backbone not in ("G", "S"):
            raise ValueError(backbone)
        if head not in ("IRM", "CRM", "DF1", "DF3", "DF5"):
            raise ValueError(head)
        self.backbone, self.head_name = backbone, head
        self.s_mode, self.s_groups, self.s_mix = s_mode, s_groups, s_mix
        self.s_gate = s_gate
        self.df_order = int(head[2:]) if head.startswith("DF") else 0
        self.bands = N_BANDS
        self.bandmap = BandMap()
        self.dsp = CausalDsp()
        self.sfe = SFE(3)
        self.enc1 = CausalConv2d(9, ch, (1, 5), stride=(1, 2), pad_freq=2, bias=True)
        self.enc2 = CausalConv2d(ch, ch, (1, 5), stride=(1, 2), groups=2, pad_freq=2, bias=True)
        self.tf = nn.ModuleList([SepTF(ch, 1 + i) for i in range(2)])
        if backbone == "G":
            self.bottleneck = GBlock(ch, 33, inter, hidden)
        else:
            if self.s_mode == "g":
                self.bottleneck = nn.ModuleList([
                    SBlockG(ch, s_hidden, groups=s_groups, dense_mix=s_mix,
                            gate=s_gate)
                    for _ in range(blocks)])
            else:
                self.bottleneck = nn.ModuleList([SBlock(ch, s_hidden)
                                                 for _ in range(blocks)])
        self.up1 = nn.ConvTranspose2d(ch, ch, (1, 5), stride=(1, 2),
                                      padding=(0, 2), output_padding=(0, 0))
        self.up2 = nn.ConvTranspose2d(ch, ch, (1, 5), stride=(1, 2),
                                      padding=(0, 2), output_padding=(0, 0))
        self.pw1 = nn.Conv2d(ch, ch, 1)
        self.pw2 = nn.Conv2d(ch, ch, 1)
        self.act = nn.PReLU(ch)
        out_ch = {"IRM": 1, "CRM": 2}.get(head, 1 + 2 * self.df_order)
        self.head = nn.Conv2d(ch, out_ch, 1)
        self._init_step0()

    # ------------------------------------------------------------------
    def _init_step0(self):
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            if self.head_name == "IRM":
                self.head.bias.fill_(math.log(0.99 / 0.01))
            elif self.head_name == "CRM":
                self.head.bias[0] = math.atanh(0.99 / 2)
                self.head.bias[1] = 0.0
            else:
                # gain 0.5 + sigmoid(g) = 0.99; current tap implicit 1;
                # history taps tanh(0)=0.
                self.head.bias[0] = math.log(0.49 / 0.51)
                self.head.bias[1:] = 0.0

    def init_state(self, batch, device=None):
        if self.backbone == "G":
            return [torch.zeros(1, batch, self.bottleneck.gru.hidden_size,
                                device=device)]
        h = self.bottleneck[0].time.hidden_size
        return [torch.zeros(1, batch * 33, h, device=device)
                for _ in self.bottleneck]

    def _body(self, feat, state):
        return self._body_bands(self.bandmap.bm(feat), state)

    def _body_bands(self, x, state):
        taps = {}
        x = self.sfe(x)                              # [B,9,T,129]
        x = self.act(self.enc1(x))                   # [B,C,T,65]
        skip1 = x
        x = self.act(self.enc2(x))                   # [B,C,T,33]
        taps["mid"] = x
        for blk in self.tf:
            x = blk(x)
        if self.backbone == "G":
            st = state[0] if isinstance(state, (list, tuple)) else state
            x, state = self.bottleneck(x, st)
        else:
            new = []
            for blk, s in zip(self.bottleneck, state or [None] * len(self.bottleneck)):
                x, s = blk(x, s)
                new.append(s)
            state = new
        taps["bottleneck"] = x
        x = self.act(self.pw1(self.up1(x)) + skip1)  # [B,C,T,65]
        x = self.act(self.pw2(self.up2(x)))          # [B,C,T,129]
        y = self.head(x)                             # [B,out,T,129]
        return y.permute(0, 2, 3, 1), state, taps    # [B,T,129,out]

    def forward(self, spec, state=None, return_features=False):
        """spec complex [B,T,257] -> enhanced complex [B,T,257]."""
        y, state, taps = self._body(features_from_spec(spec), state)
        out = self.apply_head(y, spec)
        state = [state] if torch.is_tensor(state) else state
        return (out, state, taps) if return_features else (out, state)

    def forward_bands(self, bands, re, im, state=None, hist=None,
                      return_hist=False):
        """Deployment entry: firmware supplies 129-band features and spectra."""
        if re.ndim == 2:
            re, im = re[:, None, :], im[:, None, :]
            bands = bands[:, :, None, :] if bands.ndim == 3 else bands
        y, state, _ = self._body_bands(bands, state)
        o_re, o_im, new_hist = self.apply_head_real(y, re, im, hist)
        if return_hist and new_hist is None:
            cur = [(re[:, i], im[:, i]) for i in range(re.shape[1])]
            new_hist = (cur[::-1] + (list(hist) if hist else []))[: self.df_order]
        if return_hist:
            return o_re, o_im, state, new_hist
        return o_re, o_im, state

    def forward_real(self, re, im, state=None, hist=None, return_hist=False):
        """Real-valued twin for ONNX/ESP-DL; no complex dtype anywhere."""
        if re.ndim == 2:
            re, im = re[:, None, :], im[:, None, :]
        mag = torch.sqrt(re.square() + im.square() + 1e-12)
        feat = torch.stack([torch.log1p(mag), re / mag.clamp(min=1e-6),
                            im / mag.clamp(min=1e-6)], dim=1)
        y, state, _ = self._body(feat, state)
        o_re, o_im, new_hist = self.apply_head_real(y, re, im, hist)
        if return_hist and new_hist is None:
            cur = [(re[:, i], im[:, i]) for i in range(re.shape[1])]
            new_hist = (cur[::-1] + (list(hist) if hist else []))[: self.df_order]
        if return_hist:
            return o_re, o_im, state, new_hist
        return o_re, o_im, state

    def head_mask(self, y):
        """Bounded deployment output: no spectrum multiply inside the graph."""
        if self.head_name == "IRM":
            return torch.sigmoid(y[..., 0])
        if self.head_name == "CRM":
            return 2 * torch.tanh(y)
        gain = 0.5 + torch.sigmoid(y[..., 0])
        taps = 0.5 * torch.tanh(y[..., 1:])
        return torch.cat([gain.unsqueeze(-1), taps], dim=-1)

    def apply_mask_real(self, mask, re, im, hist=None):
        if self.head_name == "IRM":
            m = self.bandmap.bs(mask)
            return re * m, im * m, hist
        if self.head_name == "CRM":
            mr = self.bandmap.bs(mask[..., 0])
            mi = self.bandmap.bs(mask[..., 1])
            return re * mr - im * mi, re * mi + im * mr, hist
        gain = self.bandmap.bs(mask[..., 0])
        k = self.df_order
        out_re, out_im = re, im
        t = re.shape[1]
        for j in range(k):
            cr = self.bandmap.bs(mask[..., 1 + j])
            ci = self.bandmap.bs(mask[..., 1 + k + j])
            prev_re = torch.zeros_like(re)
            prev_im = torch.zeros_like(im)
            if t > j + 1:
                prev_re[:, j + 1:] = re[:, : t - (j + 1)]
                prev_im[:, j + 1:] = im[:, : t - (j + 1)]
            for i in range(min(j + 1, t)):
                h_idx = j - i
                if hist is not None and h_idx < len(hist):
                    prev_re[:, i], prev_im[:, i] = hist[h_idx]
            out_re = out_re + cr * prev_re - ci * prev_im
            out_im = out_im + cr * prev_im + ci * prev_re
        return gain * out_re, gain * out_im, hist

    def forward_mask(self, bands, state=None):
        y, state, _ = self._body_bands(bands, state)
        state = [state] if torch.is_tensor(state) else state
        return self.head_mask(y), state

    def apply_head_real(self, y, re, im, hist=None):
        if self.head_name == "IRM":
            mask = self.bandmap.bs(torch.sigmoid(y[..., 0]))
            return re * mask, im * mask, hist
        if self.head_name == "CRM":
            m = 2 * torch.tanh(y)
            mr = self.bandmap.bs(m[..., 0])
            mi = self.bandmap.bs(m[..., 1])
            return re * mr - im * mi, re * mi + im * mr, hist
        gain = self.bandmap.bs(0.5 + torch.sigmoid(y[..., 0]))
        taps = 0.5 * torch.tanh(y[..., 1:])
        k = self.df_order
        out_re, out_im = re, im
        t = re.shape[1]
        for j in range(k):
            cr = self.bandmap.bs(taps[..., j])
            ci = self.bandmap.bs(taps[..., k + j])
            prev_re = torch.zeros_like(re)
            prev_im = torch.zeros_like(im)
            if t > j + 1:
                prev_re[:, j + 1:] = re[:, : t - (j + 1)]
                prev_im[:, j + 1:] = im[:, : t - (j + 1)]
            # the first j+1 frames must come from the carried history
            for i in range(min(j + 1, t)):
                h_idx = j - i
                if hist is not None and h_idx < len(hist):
                    prev_re[:, i], prev_im[:, i] = hist[h_idx]
            out_re = out_re + cr * prev_re - ci * prev_im
            out_im = out_im + cr * prev_im + ci * prev_re
        new_hist = None
        if hist is not None:
            cur = [(re[:, i], im[:, i]) for i in range(t)]
            if t >= k:
                new_hist = cur[-k:][::-1]
            else:
                new_hist = (cur[::-1] + list(hist))[:k]
        return gain * out_re, gain * out_im, new_hist

    def apply_head(self, y, spec):
        if self.head_name == "IRM":
            mask = torch.sigmoid(y[..., 0])
            mask = self.bandmap.bs(mask)
            return spec * mask
        if self.head_name == "CRM":
            m = 2 * torch.tanh(y)
            re = self.bandmap.bs(m[..., 0])
            im = self.bandmap.bs(m[..., 1])
            return spec * torch.complex(re.to(spec.real.dtype),
                                        im.to(spec.real.dtype))
        # DF-K: band gain times (current + causal residual taps)
        gain = 0.5 + torch.sigmoid(y[..., 0])
        gain = self.bandmap.bs(gain)                 # [B,T,257]
        taps = 0.5 * torch.tanh(y[..., 1:])          # [B,T,129,2K]
        k = self.df_order
        out = spec
        for j in range(k):
            re = self.bandmap.bs(taps[..., j])       # [B,T,257]
            im = self.bandmap.bs(taps[..., k + j])
            prev = F.pad(spec[:, : spec.shape[1] - (j + 1), :],
                         (0, 0, j + 1, 0))
            c = torch.complex(re.to(spec.real.dtype), im.to(spec.real.dtype))
            out = out + c * prev
        return gain * out

    # ------------------------------------------------------------------
    def param_counts(self):
        learned = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fixed = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        fixed += self.bandmap.hi.numel() + self.bandmap.hi_t.numel()
        return {"P_model": learned, "P_fixed": fixed,
                "P_graph_total": learned + fixed}

    def signature(self):
        return {"family": "cap2", "backbone": self.backbone,
                "head": self.head_name, "n_fft": NFFT, "hop": HOP,
                "s_mode": self.s_mode, "s_groups": self.s_groups,
                "s_mix": self.s_mix, "s_gate": self.s_gate}


class Cap2Bundle:
    """Trainer bundle: PCM in, PCM out, same loss interface as E0/E1."""

    def __init__(self, cfg, device):
        self.model = Cap2Net(**{k: cfg[k] for k in
                                ("backbone", "head", "ch", "inter", "hidden",
                                 "s_hidden", "blocks", "df_order", "s_mode",
                                 "s_groups", "s_mix", "s_gate")
                                if k in cfg}).to(device)
        self.dsp = self.model.dsp
        self.delay = 0
        self.period = 1
        self.chunks = None

    def forward_audio(self, mix16, return_features=False):
        spec = self.dsp.analysis(mix16)
        if return_features:
            out, _, taps = self.model(spec, return_features=True)
            return self.dsp.render(out, length=mix16.shape[-1]), taps
        out, _ = self.model(spec)
        return self.dsp.render(out, length=mix16.shape[-1])

    def signature(self):
        return self.model.signature()

    def student_init(self, cfg):
        return {"family": "cap2", **self.model.signature(),
                "params": self.model.param_counts()}
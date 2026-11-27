"""LBMCRN v1.0 waveform inference."""
from pathlib import Path
import hashlib
import json

import torch
from torch import nn

from _backbone import NS3STNet

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints"


class LBMCRN(nn.Module):
    """Low-rank band-mixing convolutional recurrent network for 16 kHz audio."""

    def __init__(self):
        super().__init__()
        self.backbone = NS3STNet("S-M")

    @classmethod
    def from_pretrained(cls, checkpoint_dir=CHECKPOINTS, device="cpu"):
        folder = Path(checkpoint_dir)
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "lbmcrn-v1.0.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        model.backbone.load_state_dict(state["model"], strict=True)
        return model.to(device).eval()

    def forward(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform[None]
        if waveform.ndim != 2 or waveform.shape[-1] == 0 or waveform.shape[0] == 0:
            raise ValueError("Expected a nonempty [samples] or [batch, samples] waveform")
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains NaN or infinity")
        spectrum = self.backbone.dsp.analysis(waveform)
        enhanced, _ = self.backbone(spectrum)
        return self.backbone.dsp.render(enhanced, length=waveform.shape[-1])

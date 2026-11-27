"""Enhance a WAV file or directory with LBMCRN v1.0."""
import argparse
from pathlib import Path

import soundfile as sf
import torch
from lbmcrn import LBMCRN, CHECKPOINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("Input does not exist")
    files = sorted(args.input.glob("*.wav")) if args.input.is_dir() else [args.input]
    if not files:
        parser.error("No WAV files found")
    torch.set_num_threads(1)
    model = LBMCRN.from_pretrained(args.checkpoint_dir, args.device)
    for path in files:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != 16000 or audio.shape[1] != 1:
            raise ValueError(f"{path}: expected mono 16 kHz audio")
        output = args.output / path.name if args.input.is_dir() else args.output
        if output.resolve() == path.resolve():
            raise ValueError("Output must differ from input")
        with torch.inference_mode():
            waveform = model(torch.from_numpy(audio[:, 0]).to(args.device))[0].cpu().numpy()
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, waveform, sr, subtype="PCM_16")
        print(output)


if __name__ == "__main__":
    main()

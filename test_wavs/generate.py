"""Generate a deterministic synthetic smoke-test WAV."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    time = np.arange(32000, dtype=np.float32) / 16000
    clean = .08 * np.sin(2 * np.pi * 220 * time)
    noise = np.random.default_rng(43).normal(0, .025, len(time)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, clean + noise, 16000, subtype="PCM_16")

if __name__ == "__main__":
    main()

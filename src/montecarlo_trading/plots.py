from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_terminal_histograms(terminal_by_name: Dict[str, np.ndarray], outdir: str) -> None:
    ensure_dir(outdir)
    for name, W in terminal_by_name.items():
        plt.figure()
        plt.hist(W, bins=60)
        plt.title(f"Terminal Wealth Distribution: {name}")
        plt.xlabel("Terminal Wealth")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"terminal_hist_{name}.png"), dpi=160)
        plt.close()


def plot_sample_paths(paths_by_name: Dict[str, Optional[np.ndarray]], outdir: str) -> None:
    ensure_dir(outdir)
    for name, paths in paths_by_name.items():
        if paths is None:
            continue
        plt.figure()
        # Plot up to 50 paths
        for i in range(paths.shape[0]):
            plt.plot(paths[i])
        plt.title(f"Sample Wealth Paths: {name}")
        plt.xlabel("Round")
        plt.ylabel("Wealth")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"paths_{name}.png"), dpi=160)
        plt.close()

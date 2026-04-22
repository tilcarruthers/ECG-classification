from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecg_repo.utils.io import ensure_dir


def save_confusion_matrix_figure(confusion: np.ndarray, labels: list[str], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha='center', va='center')

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_LABELS = {
    0: "NOR",
    1: "SVEB",
    2: "VEB",
    3: "UNK",
}


@dataclass(frozen=True)
class LabelPolicy:
    allowed_labels: tuple[int, ...] = (0, 1, 2)
    dropped_labels: tuple[int, ...] = (3,)

    def keep(self, label: int) -> bool:
        return label in self.allowed_labels

    def map_label(self, label: int) -> int:
        if label not in self.allowed_labels:
            raise ValueError(f"Label {label} is not allowed under this policy.")
        return label


def label_name(label: int) -> str:
    return CANONICAL_LABELS.get(int(label), f"UNKNOWN_{label}")


def collapse_normal_vs_abnormal(label: int) -> int:
    return 0 if int(label) == 0 else 1

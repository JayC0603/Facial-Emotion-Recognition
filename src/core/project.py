from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    config: dict[str, Any]

    @classmethod
    def load(cls, start: Path | None = None) -> "ProjectContext":
        start_path = (start or Path.cwd()).resolve()

        for candidate in (start_path, *start_path.parents):
            config_path = candidate / "config.yaml"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                return cls(root=candidate, config=config)

        raise FileNotFoundError(
            "Cannot locate FER_Pipeline/config.yaml from current directory."
        )

    def resolve(self, relative_path: str | Path) -> Path:
        return (self.root / relative_path).resolve()

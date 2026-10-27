"""Configuration for a declared, isolated scientific run."""

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import json

DEFAULT_SOURCES = ("CSI-maize_podaci/IMAGES", "Stress1_13das", "Stress2_19das")


def relative_input(value: str) -> str:
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Source folder must be a relative contained path: {value}")
    if path.parts[0] in {
        "prototype",
        "research_audit",
        ".git",
        "runs",
        "runtime",
        "metadata",
        "reports",
        "logs",
    }:
        raise ValueError(f"Source folder overlaps generated files: {value}")
    return path.as_posix()


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    work_dir: Path
    source_folders: tuple[str, ...] = DEFAULT_SOURCES
    workbook_folders: tuple[str, ...] = ()
    models_root: Path | None = None
    mode: str = "raw"
    profile: str = "publication"
    device: str = "auto"
    seed: int = 20261024
    expected_acquisitions: int | None = 41
    zenodo_record_url: str | None = None

    def __post_init__(self):
        for key in ("data_root", "work_dir", "models_root"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, Path(value).expanduser().absolute())
        folders = tuple(relative_input(x) for x in self.source_folders)
        if len(set(folders)) != len(folders):
            raise ValueError("Duplicate source folder")
        for index, folder in enumerate(folders):
            logical = PurePosixPath(folder)
            if any(
                logical.is_relative_to(PurePosixPath(other))
                or PurePosixPath(other).is_relative_to(logical)
                for other in folders[index + 1 :]
            ):
                raise ValueError("Source folder bindings overlap")
            if self.work_dir.resolve().is_relative_to((self.data_root / folder).resolve()):
                raise ValueError("The work directory cannot live inside a bound raw source")
        object.__setattr__(self, "source_folders", folders)
        books = tuple(relative_input(x) for x in self.workbook_folders)
        if len(set(books)) != len(books):
            raise ValueError("Duplicate workbook folder")
        object.__setattr__(self, "workbook_folders", books)
        if self.data_root == self.work_dir:
            raise ValueError("The work directory must be separate from the raw data")
        if self.mode not in {"raw", "example", "replay"}:
            raise ValueError("mode must be raw, example or replay")
        if self.profile not in {"publication", "learning", "example"}:
            raise ValueError("Unknown workflow profile")

    def to_dict(self):
        return {
            k: str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
            for k, v in asdict(self).items()
        }

    @classmethod
    def load(cls, path: Path):
        path = Path(path).absolute()
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        for key in ("data_root", "work_dir", "models_root"):
            if values.get(key) is not None:
                p = Path(values[key]).expanduser()
                values[key] = p if p.is_absolute() else path.parent / p
        if "source_folders" in values:
            values["source_folders"] = tuple(values["source_folders"])
        if "workbook_folders" in values:
            values["workbook_folders"] = tuple(values["workbook_folders"])
        return cls(**values)

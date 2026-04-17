from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    """Centralized application settings and project paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir_name: str = "data"
    raw_dir_name: str = "raw"
    processed_dir_name: str = "processed"
    models_dir_name: str = "models"
    artifacts_dir_name: str = "artifacts"
    assets_dir_name: str = "assets"
    basics_filename: str = "title.basics.tsv"
    ratings_filename: str = "title.ratings.tsv"
    prepared_parquet_name: str = "imdb_prepared.parquet"
    prepared_pickle_name: str = "imdb_prepared.pkl"
    default_chunksize: int = 200_000
    default_min_votes: int = 500
    default_success_rating: float = 7.0
    default_success_votes: int = 25_000
    random_seed: int = 42
    target_title_types: tuple[str, ...] = (
        "movie",
        "tvSeries",
        "tvMiniSeries",
        "tvMovie",
        "tvSpecial",
        "video",
    )

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / self.raw_dir_name

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / self.processed_dir_name

    @property
    def models_dir(self) -> Path:
        return self.project_root / self.models_dir_name

    @property
    def model_artifacts_dir(self) -> Path:
        return self.models_dir / self.artifacts_dir_name

    @property
    def assets_dir(self) -> Path:
        return self.project_root / self.assets_dir_name

    @property
    def prepared_parquet_path(self) -> Path:
        return self.processed_data_dir / self.prepared_parquet_name

    @property
    def prepared_pickle_path(self) -> Path:
        return self.processed_data_dir / self.prepared_pickle_name

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.model_artifacts_dir,
            self.assets_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = AppSettings()

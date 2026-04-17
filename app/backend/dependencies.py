from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.services.data_loader import load_dataset


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    return load_dataset()

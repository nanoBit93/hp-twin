import pandas as pd
import pytest

from hp_twin.adapters.io_csv import CsvSchemaError, validate_frame
from scripts_helper import synthetic_frame


def test_synthetic_frame_valid() -> None:
    df = validate_frame(synthetic_frame(days=1))
    assert str(df.index.tz) == "UTC"


def test_missing_column_rejected() -> None:
    with pytest.raises(CsvSchemaError):
        validate_frame(synthetic_frame(days=1).drop(columns=["p_el_w"]))


def test_gap_rejected() -> None:
    df = synthetic_frame(days=1).drop(index=[10])
    with pytest.raises(CsvSchemaError):
        validate_frame(pd.DataFrame(df))

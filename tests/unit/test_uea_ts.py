from pathlib import Path

import numpy as np
import pytest

from nyquistguard.data.uea_ts import ChannelStandardizer, load_uea_ts


def _write_tiny_ts(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "@problemName Tiny",
                "@timeStamps false",
                "@missing false",
                "@univariate false",
                "@dimensions 2",
                "@equalLength true",
                "@seriesLength 3",
                "@classLabel true A B",
                "@data",
                "1,2,3:4,5,6:A",
                "2,3,4:5,6,7:B",
            ]
        ),
        encoding="utf-8",
    )


def test_load_uea_ts_uses_case_channel_time_layout(tmp_path: Path):
    path = tmp_path / "Tiny_TRAIN.ts"
    _write_tiny_ts(path)
    collection = load_uea_ts(path, split="train")
    assert collection.x.shape == (2, 2, 3)
    assert collection.x.dtype == np.float32
    assert collection.class_names == ("A", "B")
    assert collection.sample_ids == ("Tiny:train:00000", "Tiny:train:00001")


def test_channel_standardizer_is_train_only(tmp_path: Path):
    path = tmp_path / "Tiny.ts"
    _write_tiny_ts(path)
    train = load_uea_ts(path, split="train")
    test = load_uea_ts(path, split="test")
    standardizer = ChannelStandardizer.fit(train)
    transformed = standardizer.transform(train.x)
    assert np.allclose(transformed.mean(axis=(0, 2)), 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="training split"):
        ChannelStandardizer.fit(test)

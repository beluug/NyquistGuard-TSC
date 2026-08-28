from collections import OrderedDict

import pytest
import torch

from nyquistguard.experiments.v3_weight_average import average_state_dicts


def test_fixed_weight_average_blends_floating_tensors() -> None:
    first = OrderedDict(weight=torch.tensor([1.0, 3.0]))
    second = OrderedDict(weight=torch.tensor([3.0, 1.0]))
    result = average_state_dicts(first, second, 0.5, 0.5)
    assert result["weight"].tolist() == pytest.approx([2.0, 2.0])


def test_fixed_weight_average_rejects_mismatched_states() -> None:
    with pytest.raises(ValueError):
        average_state_dicts(
            OrderedDict(a=torch.tensor([1.0])),
            OrderedDict(b=torch.tensor([1.0])),
            0.5,
            0.5,
        )

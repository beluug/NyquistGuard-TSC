import math

import torch

from nyquistguard.losses.monotonicity import AcceptanceMonotonicityLoss
from nyquistguard.losses.selective_risk import SelectiveRiskLoss
from nyquistguard.models.nyquist_gate import NyquistObservabilityGate
from nyquistguard.models.nyquistguard_tsc import NyquistGuardTSC
from nyquistguard.models.physical_filterbank import PhysicalGaborFilterBank
from nyquistguard.models.selective_head import SelectiveHead


def _sine(rate: float, frequency: float, phase: float, duration: float) -> torch.Tensor:
    time = torch.arange(round(rate * duration), dtype=torch.float32) / rate
    return torch.sin(2.0 * math.pi * frequency * time + phase)


def _low_frequency_dataset(rates, phases, duration=1.5):
    records = []
    for phase_id, phase in enumerate(phases):
        for class_id, frequency in enumerate((3.0, 7.0)):
            for rate in rates:
                signal = _sine(rate, frequency, phase, duration)
                # A deterministic small nuisance component prevents a completely
                # trivial single-bin lookup while remaining below every Nyquist.
                signal = signal + 0.05 * _sine(rate, 1.0, phase * 0.5, duration)
                records.append((signal, float(rate), class_id, phase_id))
    maximum_length = max(signal.numel() for signal, *_ in records)
    x = torch.zeros(len(records), 1, maximum_length)
    mask = torch.ones(len(records), maximum_length, dtype=torch.bool)
    sampling_rates = torch.empty(len(records))
    targets = torch.empty(len(records), dtype=torch.long)
    source_ids = torch.empty(len(records), dtype=torch.long)
    for index, (signal, rate, target, phase_id) in enumerate(records):
        x[index, 0, : signal.numel()] = signal
        mask[index, : signal.numel()] = False
        sampling_rates[index] = rate
        targets[index] = target
        source_ids[index] = 2 * phase_id + target
    return x, mask, sampling_rates, targets, source_ids


def test_multiple_physical_frequencies_have_rate_consistent_responses():
    bank = PhysicalGaborFilterBank(
        1,
        8,
        min_center_hz=1.0,
        max_center_hz=9.0,
        min_sigma_seconds=0.07,
        max_sigma_seconds=0.18,
        max_kernel_seconds=1.0,
    )
    errors = {}
    for frequency in (2.0, 5.0, 8.0):
        responses = []
        winners = []
        for rate in (24.0, 40.0, 80.0):
            signal = _sine(rate, frequency, 0.3, 3.0).view(1, 1, -1)
            output = bank(signal, rate)
            edge = int(0.55 * rate)
            response = output[0, :, edge:-edge].mean(dim=-1)
            responses.append(response)
            winners.append(int(response.argmax()))
        stacked = torch.stack(responses)
        winning_response = stacked[:, winners[0]]
        relative_error = (
            (winning_response.max() - winning_response.min())
            / winning_response.mean().clamp_min(1e-6)
        )
        errors[frequency] = float(relative_error.detach())
        assert len(set(winners)) == 1
        assert errors[frequency] < 0.10
    print({"physical_frequency_relative_errors": errors})


def test_unseen_rate_low_frequency_classification_is_stable():
    torch.manual_seed(17)
    train = _low_frequency_dataset(
        rates=(60.0, 45.0, 30.0),
        phases=(0.0, 0.7, 1.4, 2.1),
    )
    test = _low_frequency_dataset(
        rates=(54.0, 36.0, 24.0, 18.0),
        phases=(0.2, 0.9, 1.6, 2.3),
    )
    model = NyquistGuardTSC(
        input_channels=1,
        num_classes=2,
        num_bands=8,
        pooled_positions=16,
        hidden_dim=24,
        encoder_depth=2,
        dropout=0.0,
        min_center_hz=1.0,
        max_center_hz=9.0,
        min_sigma_seconds=0.05,
        max_sigma_seconds=0.18,
        max_kernel_seconds=0.8,
        reference_sampling_rate_hz=60.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x, mask, rates, targets, _ = train
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, rates, mask)
        loss = torch.nn.functional.cross_entropy(output["logits"], targets)
        loss.backward()
        optimizer.step()
        if float(loss.detach()) < 0.01:
            break

    model.eval()
    with torch.inference_mode():
        x, mask, rates, targets, source_ids = test
        output = model(x, rates, mask)
        predictions = output["logits"].argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
        flip_count = 0
        comparisons = 0
        for source_id in torch.unique(source_ids):
            group = predictions[source_ids == source_id]
            flip_count += int(torch.count_nonzero(group != group[0]))
            comparisons += group.numel() - 1
        flip_rate = flip_count / max(comparisons, 1)
    print({"unseen_rate_accuracy": float(accuracy), "prediction_flip_rate": flip_rate})
    assert accuracy >= 0.95
    assert flip_rate <= 0.05


def test_selective_head_rejects_information_insufficient_low_rate_views():
    """High-band class evidence is absent in the synthetic low-rate views.

    The classifier loss is intentionally high for the low-rate half and low for
    the high-rate half.  This isolates whether gate/rate-conditioned selection,
    coverage, and pair monotonicity learn the physically correct allocation.
    """

    torch.manual_seed(42)
    batch_size = 16
    centers = torch.tensor([2.0, 8.0, 14.0, 18.0])
    sigmas = torch.tensor([0.12, 0.08, 0.05, 0.04])
    gate_fn = NyquistObservabilityGate()
    high_rates = torch.full((batch_size,), 64.0)
    low_rates = torch.full((batch_size,), 20.0)
    high_gates = gate_fn(high_rates, centers, sigmas)
    low_gates = gate_fn(low_rates, centers, sigmas)
    embeddings = torch.zeros(batch_size, 8)
    entropy_high = torch.full((batch_size,), 0.05)
    entropy_low = torch.full((batch_size,), 0.95)
    targets = torch.arange(batch_size) % 2
    logits_high = torch.full((batch_size, 2), -3.0)
    logits_high[torch.arange(batch_size), targets] = 3.0
    logits_low = torch.zeros(batch_size, 2)

    head = SelectiveHead(8, 4, hidden_dim=16, dropout=0.0, reference_sampling_rate_hz=64.0)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-2)
    risk = SelectiveRiskLoss(target_coverage=0.5, coverage_weight=20.0)
    monotonicity = AcceptanceMonotonicityLoss()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        q_high = torch.sigmoid(head(embeddings, high_gates, high_rates, entropy_high))
        q_low = torch.sigmoid(head(embeddings, low_gates, low_rates, entropy_low))
        result = risk(
            torch.cat([logits_high, logits_low]),
            torch.cat([targets, targets]),
            torch.cat([q_high, q_low]),
        )
        loss = result["total"] + monotonicity(q_low, q_high)
        loss.backward()
        optimizer.step()

    with torch.inference_mode():
        q_high = torch.sigmoid(head(embeddings, high_gates, high_rates, entropy_high)).mean()
        q_low = torch.sigmoid(head(embeddings, low_gates, low_rates, entropy_low)).mean()
    print({"accept_high_rate": float(q_high), "accept_low_rate": float(q_low)})
    assert q_high > 0.85
    assert q_low < 0.15
    assert q_high - q_low > 0.75


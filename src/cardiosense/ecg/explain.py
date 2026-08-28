"""Integrated Gradients for the ECG models.

This is **Integrated Gradients**, and it is called that. It is not attention:
neither :class:`~cardiosense.ecg.models.ECGCNN1D` nor
:class:`~cardiosense.ecg.models.ECGResNet1D` contains an attention mechanism, and
labelling a saliency map "attention" would be a straightforward misdescription of
the architecture.

How Integrated Gradients works
------------------------------

A plain input gradient tells you the sensitivity of the output at one point,
which is unreliable when the network saturates — a feature can be decisive and
still have a near-zero local gradient. IG instead integrates the gradient along a
straight path from a baseline to the actual input::

    IG_i(x) = (x_i - baseline_i) * integral_0^1 dF(baseline + a(x - baseline))/dx_i da

approximated with a Riemann sum over ``n_steps`` points.

Its defining property is **completeness**: the attributions sum to
``F(x) - F(baseline)``. :func:`integrated_gradients` returns the convergence
error so this can be checked rather than assumed — a large error means
``n_steps`` is too small and the attribution map should not be trusted.

The baseline is an all-zero signal. After per-lead z-scoring, zero is the mean of
each lead, so the baseline reads as "a flat trace with no deflection" — the
natural reference for "what did this deflection contribute?".

Limitations, which belong in the report
---------------------------------------

1. **Attribution is not diagnosis.** A peak over the ST segment means the model's
   output was sensitive to samples there. It does not mean the model measured ST
   elevation, and it does not mean ST elevation is present.
2. **The baseline choice changes the answer.** A zero baseline treats absence of
   deflection as neutral. A different baseline (a mean normal ECG, say) would
   redistribute attributions, and there is no uniquely correct choice.
3. **Attributions are per-sample, not per-beat.** IG will happily place a sharp
   peak on one sample of a QRS complex; the physiologically meaningful unit is
   the whole complex. Smoothing before plotting makes this readable but is a
   presentation choice, not extra evidence.
4. **Correlated leads share credit arbitrarily.** Leads II, III and aVF view
   overlapping territory; which one receives attribution can be close to
   arbitrary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..common.config import Config
from ..common.io_utils import save_json
from ..common.logging_utils import get_logger
from ..common.paths import ensure_dir
from ..common.plots import save_figure

__all__ = ["integrated_gradients", "explain_records", "plot_attribution", "run_ig_analysis"]

logger = get_logger("ecg.explain")


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    n_steps: int = 128,
    use_captum: bool = True,
    tolerance: float = 0.05,
    auto_refine: bool = True,
    max_steps: int = 1024,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute Integrated Gradients attributions for one target class.

    Uses Captum when available (battle-tested), otherwise an explicit
    implementation. Both are checked against the completeness property.

    **On the completeness check.** Attributions should sum to
    ``F(x) - F(baseline)``. Reporting only a *relative* error is misleading: when
    the model is near-indifferent about a record, ``F(x) - F(baseline)`` is close
    to zero and the relative error explodes even though the attributions are
    numerically fine. So both absolute and relative error are computed, and a
    warning fires only when **both** exceed their tolerances. When ``auto_refine``
    is set, ``n_steps`` is doubled (up to ``max_steps``) and the attribution
    recomputed rather than returning a map that is known not to sum correctly.

    Args:
        model: The trained model, in eval mode.
        inputs: Shape ``(batch, n_leads, n_samples)``.
        target_class: Index of the class to explain.
        baseline: Reference input; defaults to zeros (a flat trace).
        n_steps: Riemann-sum steps.
        use_captum: Prefer Captum if it is installed.
        tolerance: Relative convergence-error tolerance.
        auto_refine: Retry with more steps when the check fails.
        max_steps: Ceiling for refinement.

    Returns:
        ``(attributions, info)`` with attributions shaped like ``inputs``.
    """
    model.eval()
    inputs = inputs.detach()
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    with torch.no_grad():
        expected = (model(inputs)[:, target_class]
                    - model(baseline)[:, target_class]).cpu().numpy()

    steps = int(n_steps)
    attempts: list[dict[str, Any]] = []

    while True:
        attributions, implementation = _compute_ig(
            model, inputs, baseline, target_class, steps, use_captum
        )
        actual = attributions.sum(dim=(1, 2)).detach().cpu().numpy()

        absolute_error = float(np.mean(np.abs(actual - expected)))
        relative_error = float(np.mean(np.abs(actual - expected)
                                       / np.maximum(np.abs(expected), 1e-6)))
        attempts.append({"n_steps": steps,
                         "absolute_error": round(absolute_error, 6),
                         "relative_error": round(relative_error, 5)})

        converged = relative_error <= tolerance or absolute_error <= 1e-2
        if converged or not auto_refine or steps >= max_steps:
            break
        steps = min(steps * 2, max_steps)
        logger.info("IG completeness not met (rel %.3f, abs %.4f); retrying with %d steps.",
                    relative_error, absolute_error, steps)

    info = {
        "implementation": implementation,
        "n_steps": steps,
        "target_class": int(target_class),
        "completeness_expected": np.round(expected, 5).tolist(),
        "completeness_actual": np.round(actual, 5).tolist(),
        "mean_absolute_convergence_error": round(absolute_error, 6),
        "mean_relative_convergence_error": round(relative_error, 5),
        "converged": bool(converged),
        "attempts": attempts,
    }

    if not converged:
        logger.warning(
            "IG did not converge after %d steps (relative error %.3f, absolute %.4f). "
            "Treat these attributions as indicative only.",
            steps, relative_error, absolute_error,
        )
    return attributions.detach().cpu().numpy(), info


def _compute_ig(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    n_steps: int,
    use_captum: bool,
) -> tuple[torch.Tensor, str]:
    """Dispatch to Captum or the explicit implementation."""
    if use_captum:
        try:
            from captum.attr import IntegratedGradients

            explainer = IntegratedGradients(model)
            return explainer.attribute(
                inputs, baselines=baseline, target=target_class, n_steps=n_steps
            ), "captum"
        except ImportError:
            pass
    return _manual_ig(model, inputs, baseline, target_class, n_steps), "manual"


def _manual_ig(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    n_steps: int,
) -> torch.Tensor:
    """Explicit Integrated Gradients, used when Captum is unavailable.

    Riemann midpoint rule: sample the path at ``(i + 0.5) / n_steps`` rather than
    ``i / n_steps``, which converges faster than the left-endpoint rule for the
    same number of steps.
    """
    difference = inputs - baseline
    accumulated = torch.zeros_like(inputs)

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps
        point = (baseline + alpha * difference).clone().requires_grad_(True)
        output = model(point)[:, target_class].sum()
        gradient, = torch.autograd.grad(output, point)
        accumulated = accumulated + gradient.detach()

    return difference * accumulated / n_steps


def plot_attribution(
    waveform: np.ndarray,
    attribution: np.ndarray,
    lead_names: Sequence[str],
    leads_to_plot: Sequence[str],
    class_name: str,
    probability: float,
    true_label: int,
    path: Path | str,
    sampling_rate: int = 100,
    smooth_window: int = 5,
) -> Path:
    """Plot selected leads with attribution shown as background intensity.

    The waveform is drawn as a line; attribution magnitude is drawn underneath as
    a shaded band, so the trace stays readable. Attribution is smoothed with a
    short moving average because per-sample spikes are visually dominant and not
    physiologically meaningful — this is a *presentation* choice and does not add
    information.

    Args:
        waveform: Shape ``(n_leads, n_samples)``.
        attribution: Same shape.
        lead_names: All lead names in channel order.
        leads_to_plot: Which leads to show.
        class_name: Class being explained.
        probability: Model probability for that class.
        true_label: Ground truth (0/1).
        path: Output path.
        sampling_rate: For the time axis.
        smooth_window: Moving-average width in samples.
    """
    import matplotlib.pyplot as plt

    lead_index = {name: i for i, name in enumerate(lead_names)}
    selected = [name for name in leads_to_plot if name in lead_index] or list(lead_names[:3])

    time_axis = np.arange(waveform.shape[-1]) / sampling_rate
    magnitude = np.abs(attribution)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        magnitude = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), axis=-1, arr=magnitude
        )
    scale = magnitude.max() if magnitude.max() > 0 else 1.0

    fig, axes = plt.subplots(len(selected), 1, figsize=(12, 1.9 * len(selected)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, lead in zip(axes, selected):
        channel = lead_index[lead]
        trace = waveform[channel]
        weight = magnitude[channel] / scale

        ax.plot(time_axis, trace, color="black", lw=0.9, zorder=3)
        # Shade attribution as a band spanning the y-range of this lead.
        low, high = trace.min(), trace.max()
        ax.imshow(
            weight[None, :], aspect="auto", cmap="Reds", alpha=0.55,
            extent=(float(time_axis[0]), float(time_axis[-1]), float(low), float(high)),
            origin="lower", zorder=1, vmin=0, vmax=1,
        )
        ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=10)
        ax.grid(alpha=0.25, zorder=0)

    axes[-1].set_xlabel("time (s)")
    verdict = "correct" if int(probability >= 0.5) == int(true_label) else "INCORRECT"
    fig.suptitle(
        f"Integrated Gradients — class {class_name} | p = {probability:.3f} | "
        f"true = {int(true_label)} ({verdict})\n"
        "red intensity = |attribution|; shows where the model was sensitive, "
        "NOT that a finding is present",
        fontsize=10,
    )
    return save_figure(fig, path)


def explain_records(
    model: torch.nn.Module,
    waveforms: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    record_ids: Sequence[int],
    target_class: int,
    cfg: Config,
    device: torch.device,
    out_dir: Path | str,
    tag: str = "case",
) -> list[dict[str, Any]]:
    """Explain a set of records with respect to one class, and plot each.

    Returns:
        One record per explanation, listing the highest-attribution leads and the
        time window that carried most of the attribution mass.
    """
    out = ensure_dir(out_dir)
    classes = list(cfg.task.classes)
    class_name = classes[target_class]
    lead_names = list(cfg.dataset.lead_names)
    leads_to_plot = list(cfg.explainability.get("leads_to_plot", lead_names[:3]))
    sampling_rate = int(cfg.dataset.sampling_rate)

    batch = torch.from_numpy(np.ascontiguousarray(waveforms, dtype=np.float32)).to(device)
    attributions, info = integrated_gradients(
        model, batch, target_class,
        n_steps=int(cfg.explainability.get("n_steps", 128)),
    )

    records: list[dict[str, Any]] = []
    for position, record_id in enumerate(record_ids):
        attribution = attributions[position]
        waveform = waveforms[position]
        magnitude = np.abs(attribution)

        per_lead = magnitude.sum(axis=-1)
        top_leads = [
            {"lead": lead_names[i], "share": round(float(per_lead[i] / max(per_lead.sum(), 1e-9)), 4)}
            for i in np.argsort(per_lead)[::-1][:3]
        ]

        # Which 1-second window carries the most attribution?
        per_sample = magnitude.sum(axis=0)
        window = sampling_rate
        if per_sample.size >= window:
            sums = np.convolve(per_sample, np.ones(window), mode="valid")
            start = int(np.argmax(sums))
        else:
            start = 0

        filename = f"ig_{tag}_{class_name}_record{int(record_id)}.png"
        plot_attribution(
            waveform, attribution, lead_names, leads_to_plot, class_name,
            float(probabilities[position, target_class]), int(labels[position, target_class]),
            out / filename, sampling_rate=sampling_rate,
        )

        records.append({
            "record_id": int(record_id),
            "case_type": tag,
            "class": class_name,
            "probability": round(float(probabilities[position, target_class]), 4),
            "true_label": int(labels[position, target_class]),
            "top_leads": top_leads,
            "peak_window_seconds": [round(start / sampling_rate, 2),
                                    round((start + window) / sampling_rate, 2)],
            "figure": filename,
        })

    logger.info("Explained %d records for class %s (%s, %d steps, converged=%s, "
                "abs error %.4f)",
                len(records), class_name, info["implementation"], info["n_steps"],
                info["converged"], info["mean_absolute_convergence_error"])
    return records


def run_ig_analysis(
    model: torch.nn.Module,
    waveforms: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    cfg: Config,
    device: torch.device,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Explain representative correct and incorrect predictions for each class.

    For every class it picks a confident true positive and, where one exists, a
    false negative. The false negatives are the point of the exercise: seeing
    where the model looked when it missed an MI is far more informative than
    seeing where it looked when it succeeded.

    Args:
        model: Trained model.
        waveforms: Full waveform array (memory-mapped is fine).
        indices: Row positions of the split being explained, into ``waveforms``.
        labels: Split labels, aligned with ``indices``.
        probabilities: Split probabilities, aligned with ``indices``.
        thresholds: Per-class thresholds.
        cfg: ECG configuration.
        device: Device to run on.
        out_dir: ``results/ecg/explanations``.

    Returns:
        Summary dict, also written to ``ig_summary.json``.
    """
    out = ensure_dir(out_dir)
    classes = list(cfg.task.classes)
    n_examples = int(cfg.explainability.get("n_examples", 8))
    per_class_budget = max(1, n_examples // max(len(classes), 1))

    predictions = (probabilities >= np.asarray(thresholds)[None, :]).astype(int)
    all_records: list[dict[str, Any]] = []

    for class_index, class_name in enumerate(classes):
        true_positive = np.flatnonzero(
            (labels[:, class_index] == 1) & (predictions[:, class_index] == 1)
        )
        false_negative = np.flatnonzero(
            (labels[:, class_index] == 1) & (predictions[:, class_index] == 0)
        )

        selections: list[tuple[str, np.ndarray]] = []
        if true_positive.size:
            confident = true_positive[np.argsort(probabilities[true_positive, class_index])[::-1]]
            selections.append(("TP", confident[:per_class_budget]))
        if false_negative.size:
            confident_miss = false_negative[
                np.argsort(probabilities[false_negative, class_index])
            ]
            selections.append(("FN", confident_miss[:per_class_budget]))

        for tag, positions in selections:
            if positions.size == 0:
                continue
            batch_waveforms = np.stack([np.asarray(waveforms[int(indices[p])])
                                        for p in positions])
            all_records.extend(explain_records(
                model, batch_waveforms, labels[positions], probabilities[positions],
                record_ids=[int(indices[p]) for p in positions],
                target_class=class_index, cfg=cfg, device=device, out_dir=out, tag=tag,
            ))

    summary = {
        "method": "integrated_gradients",
        "note": "This is Integrated Gradients, not attention — neither architecture "
                "contains an attention mechanism.",
        "baseline": str(cfg.explainability.get("baseline", "zeros")),
        "n_steps": int(cfg.explainability.get("n_steps", 128)),
        "n_explanations": len(all_records),
        "explanations": all_records,
        "limitations": [
            "Attribution shows where the model was sensitive, not that a finding is present.",
            "A different baseline would redistribute the attributions; there is no uniquely "
            "correct choice.",
            "Attributions are per-sample; the meaningful physiological unit is the beat.",
            "Leads viewing overlapping territory (II/III/aVF) share credit arbitrarily.",
        ],
    }
    save_json(summary, out / "ig_summary.json")
    logger.info("Wrote %d IG explanations to %s", len(all_records), out)
    return summary

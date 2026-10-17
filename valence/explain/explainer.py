"""Group per-feature attributions into the PRD's explanation layers and build narrator fact sheets."""

from __future__ import annotations

from collections import defaultdict

from valence.contracts import (
    LAYERS,
    DataQuality,
    Driver,
    FaceOnly,
    FactSheet,
    Layer,
)
from valence.explain.describe import describe


def build_layers(
    contribs: dict[str, float],
    values: dict[str, float],
    *,
    tod_h: float | None = None,
    drivers_per_layer: int = 3,
) -> list[Layer]:
    """Sum attributions per layer; keep each layer's strongest narratable drivers.

    Positive attribution supports the predicted label, negative opposes it. Layers are ordered by
    contribution, strongest support first; the Facial Surface layer is kept even when it opposes, so
    a corrected mask stays visible.
    """
    per_layer: dict[str, list[Driver]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    for feature, shap in contribs.items():
        desc = describe(feature, values.get(feature), tod_h=tod_h)
        if desc is None:
            continue
        totals[desc.layer] += shap
        value = values.get(feature)
        per_layer[desc.layer].append(Driver(
            feature=feature, display=desc.display, value=None if value is None else round(float(value), 4),
            unit=desc.unit, shap=round(float(shap), 4), level=desc.level,
            effect="supports" if shap >= 0 else "opposes", scope=desc.scope,
        ))
    layers = []
    for name in LAYERS:
        if name not in per_layer:
            continue
        # One driver per sensor channel (motion_now and motion_ewm_1m would read as the same sentence).
        best: dict[str, Driver] = {}
        for drv in sorted(per_layer[name], key=lambda d: abs(d.shap), reverse=True):
            best.setdefault(channel_of(drv.feature), drv)
        drivers = list(best.values())[:drivers_per_layer]
        layers.append(Layer(name=name, contribution=round(totals[name], 4), drivers=drivers))
    return sorted(layers, key=lambda layer: layer.contribution, reverse=True)


def channel_of(feature: str) -> str:
    if feature.startswith("tod_"):
        return "tod"
    if feature.startswith("face_visible"):
        return "face_visible"
    return feature.split("_")[0]


def top_drivers(layers: list[Layer], k: int = 5, effect: str | None = "supports") -> list[Driver]:
    drivers = [d for layer in layers for d in layer.drivers if effect is None or d.effect == effect]
    return sorted(drivers, key=lambda d: abs(d.shap), reverse=True)[:k]


def build_fact_sheet(
    *,
    event_id: str,
    person: str,
    label: str,
    label_display: str,
    confidence: float,
    probabilities: dict[str, float],
    window_fill_min: float,
    n_people: int,
    face_only_label: str | None,
    face_only_confidence: float | None,
    face_valence: float | None,
    face_visible: bool,
    mask_override: bool,
    layers: list[Layer],
    face_visible_frac: float,
    lux: float | None,
    ood_features: list[str],
) -> FactSheet:
    return FactSheet(
        event_id=event_id, person=person, label=label, label_display=label_display,
        confidence=round(confidence, 4), confidence_display=f"{round(100 * confidence)}%",
        probabilities={k: round(v, 4) for k, v in probabilities.items()}, window_fill_min=round(window_fill_min, 1),
        n_people=n_people,
        face_only=FaceOnly(label=face_only_label if face_visible else None,
                           confidence=None if (face_only_confidence is None or not face_visible) else round(face_only_confidence, 4),
                           face_valence=None if (face_valence is None or not face_visible) else round(face_valence, 2)),
        face_visible=face_visible, mask_override=mask_override, layers=layers,
        data_quality=DataQuality(face_visible_frac=round(face_visible_frac, 3), lux=None if lux is None else round(lux, 1),
                                 ood_features=ood_features),
    )

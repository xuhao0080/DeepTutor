#!/usr/bin/env python3
"""Draw pairwise human-vs-LLM preference alignment as a dependency-free SVG."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
from xml.sax.saxutils import escape

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.human_alignment.common import METRIC_CODES, read_json
from benchmark.human_alignment.live_judge import (
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    parse_metric_codes,
    summarize_with_live_judge,
)
from benchmark.human_alignment.summarize_annotations import summarize_annotations

COLORS = {
    "target": "#2f9f91",
    "tie": "#cfd6e1",
    "baseline": "#e96d4f",
    "grid": "#e5e7eb",
    "axis": "#cfd6de",
    "text": "#111827",
    "muted": "#667085",
}


def _pct(value: Any) -> float:
    return max(0.0, min(100.0, 100.0 * float(value or 0.0)))


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "400",
    anchor: str = "middle",
    fill: str = COLORS["text"],
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
    )


def _rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = "white") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
    )


def _metric_segments(summary: dict[str, Any], code: str, prefix: str) -> dict[str, float]:
    metric = summary.get("metrics", {}).get(code, {}) or {}
    target = _pct(metric.get(f"{prefix}_target_preference_rate"))
    tie = _pct(metric.get(f"{prefix}_tie_rate"))
    baseline = _pct(metric.get(f"{prefix}_baseline_preference_rate"))
    total = target + tie + baseline
    if total > 100.0:
        scale = 100.0 / total
        target *= scale
        tie *= scale
        baseline *= scale
    elif total < 100.0 and total > 0:
        baseline += 100.0 - total
    return {"target": target, "tie": tie, "baseline": baseline}


def _bar(
    *,
    x: float,
    y_bottom: float,
    width: float,
    height: float,
    segments: dict[str, float],
) -> str:
    parts = []
    current_bottom = y_bottom
    for key in ["target", "tie", "baseline"]:
        seg_h = height * segments[key] / 100.0
        if seg_h <= 0:
            continue
        y = current_bottom - seg_h
        parts.append(_rect(x, y, width, seg_h, COLORS[key]))
        current_bottom = y
    target_label = round(segments["target"])
    if segments["target"] >= 12:
        label_y = y_bottom - (height * segments["target"] / 200.0) + 5
        parts.append(_svg_text(x + width / 2, label_y, str(target_label), size=13, weight="800", fill="white"))
    return "\n".join(parts)


def build_svg(summary: dict[str, Any], title: str, metric_codes: list[str] | None = None) -> str:
    metric_codes = metric_codes or list(METRIC_CODES)
    width = 1400
    height = 620
    left = 84
    right = 40
    top = 74
    bottom = 124
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_bottom = top + plot_h
    metric_gap = plot_w / len(metric_codes)
    bar_w = 42
    bar_gap = 4
    baseline_y = y_bottom

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 38, title, size=22, weight="800"),
    ]

    for tick in range(0, 101, 20):
        y = y_bottom - plot_h * tick / 100.0
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(_svg_text(left - 18, y + 5, str(tick), size=14, anchor="end"))
    y50 = y_bottom - plot_h * 0.5
    parts.append(
        f'<line x1="{left}" y1="{y50:.1f}" x2="{width - right}" y2="{y50:.1f}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1.2" stroke-dasharray="5 4" opacity=".6"/>'
    )
    parts.append(_svg_text(width - right - 4, y50 - 8, "50%", size=13, anchor="end", fill=COLORS["muted"]))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline_y + 32}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    parts.append(f'<line x1="{left}" y1="{baseline_y}" x2="{width - right}" y2="{baseline_y}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
    parts.append(
        f'<text x="28" y="{top + plot_h / 2}" text-anchor="middle" font-size="16" '
        f'font-weight="600" fill="{COLORS["text"]}" transform="rotate(-90 28 {top + plot_h / 2})">'
        "Preference share (%)</text>"
    )

    for idx, code in enumerate(metric_codes):
        center = left + metric_gap * (idx + 0.5)
        human_x = center - bar_w - bar_gap / 2
        llm_x = center + bar_gap / 2
        parts.append(
            _bar(
                x=human_x,
                y_bottom=y_bottom,
                width=bar_w,
                height=plot_h,
                segments=_metric_segments(summary, code, "human"),
            )
        )
        parts.append(
            _bar(
                x=llm_x,
                y_bottom=y_bottom,
                width=bar_w,
                height=plot_h,
                segments=_metric_segments(summary, code, "llm"),
            )
        )
        tick_x = center
        parts.append(f'<line x1="{tick_x:.1f}" y1="{baseline_y}" x2="{tick_x:.1f}" y2="{baseline_y + 7}" stroke="{COLORS["axis"]}" stroke-width="1.2"/>')
        parts.append(_svg_text(human_x + bar_w / 2, baseline_y + 24, "H", size=13, fill=COLORS["muted"]))
        parts.append(_svg_text(llm_x + bar_w / 2, baseline_y + 24, "L", size=13, fill=COLORS["muted"]))
        parts.append(_svg_text(center, baseline_y + 50, code, size=15))

    legend_y = height - 48
    legend_x = width / 2 - 260
    legend = [
        ("DeepTutor preferred", COLORS["target"]),
        ("Tie", COLORS["tie"]),
        ("Mock preferred", COLORS["baseline"]),
    ]
    for label, color in legend:
        parts.append(_rect(legend_x, legend_y - 14, 32, 16, color, stroke=color))
        parts.append(_svg_text(legend_x + 42, legend_y, label, size=14, anchor="start"))
        legend_x += 250 if label == "DeepTutor preferred" else 120
    parts.append(
        _svg_text(
            left,
            height - 22,
            "H = human majority preference; L = LLM-judge preference. Numbers inside teal bars show DeepTutor win rate.",
            size=13,
            anchor="start",
            fill=COLORS["muted"],
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pairwise human-vs-LLM preference alignment")
    parser.add_argument("--summary", default="", help="Existing human_alignment_summary.json")
    parser.add_argument("--annotations", default="", help="Completed annotation CSV/JSONL; used when --summary is omitted")
    parser.add_argument("--key", default="", help="annotation_key.json; required with --annotations")
    parser.add_argument("--package", default="", help="annotation_package.jsonl (default: next to key)")
    parser.add_argument(
        "--llm-source",
        choices=["live", "eval"],
        default="live",
        help="How to obtain LLM preference when --annotations is used (default: live)",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Live judge model")
    parser.add_argument("--judge-binding", default="", help="Override live judge provider binding; default uses existing LLM config")
    parser.add_argument("--judge-base-url", default="", help="Override live judge API base URL; default uses existing LLM config")
    parser.add_argument("--judge-api-key", default=None, help="Override live judge API key; default uses existing LLM config/env")
    parser.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY, help="Concurrent live judge calls")
    parser.add_argument("--judge-max-tokens", type=int, default=1800, help="Max tokens per live judge response")
    parser.add_argument("--judge-output", default="", help="Live judge JSON output")
    parser.add_argument("--limit-pairs", type=int, default=0, help="Debug: live judge only first N selected pairs")
    parser.add_argument("--metrics", default="", help="Comma-separated metric codes to judge/plot, e.g. SF or SF,PER")
    parser.add_argument("--judge-all-pairs", action="store_true", help="Judge every pair in annotation_package.jsonl, including pairs without human labels")
    parser.add_argument("--quiet", action="store_true", help="Suppress live judge progress logs")
    parser.add_argument("--tie-threshold", type=float, default=0.25, help="LLM score delta treated as tie")
    parser.add_argument("--output", default="", help="Output SVG path")
    parser.add_argument(
        "--title",
        default="Human vs. LLM Preference Alignment (DeepTutor vs. Mock)",
        help="Figure title",
    )
    args = parser.parse_args()
    metric_codes = parse_metric_codes(args.metrics)

    if args.summary:
        summary_path = Path(args.summary)
        summary = read_json(summary_path)
    else:
        if not args.annotations or not args.key:
            parser.error("provide either --summary or both --annotations and --key")
        key_path = Path(args.key)
        summary_path = key_path.parent / "human_alignment_summary.json"
        if args.llm_source == "live":
            package_path = Path(args.package) if args.package else key_path.parent / "annotation_package.jsonl"
            default_judge_name = "live_llm_judgments_all_pairs.json" if args.judge_all_pairs else "live_llm_judgments.json"
            judge_output_path = Path(args.judge_output) if args.judge_output else key_path.parent / default_judge_name
            summary = summarize_with_live_judge(
                annotations_path=Path(args.annotations),
                key_path=key_path,
                package_path=package_path,
                summary_output_path=summary_path,
                judge_output_path=judge_output_path,
                model=args.judge_model,
                binding=args.judge_binding or None,
                base_url=args.judge_base_url or None,
                api_key=args.judge_api_key,
                concurrency=args.judge_concurrency,
                max_tokens=args.judge_max_tokens,
                limit_pairs=args.limit_pairs,
                metric_codes=metric_codes,
                judge_all_pairs=args.judge_all_pairs,
                verbose=not args.quiet,
            )
        else:
            summary = summarize_annotations(
                annotations_path=Path(args.annotations),
                key_path=key_path,
                output_path=summary_path,
                tie_threshold=args.tie_threshold,
            )

    output_path = Path(args.output) if args.output else Path(summary_path).with_name("human_alignment_preference_alignment.svg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(summary, args.title, metric_codes), encoding="utf-8")
    print(f"Figure: {output_path}")
    if summary.get("llm_preference_source"):
        print(f"LLM preference source: {summary['llm_preference_source']}")


if __name__ == "__main__":
    main()

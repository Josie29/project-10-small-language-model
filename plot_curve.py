from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from slm.reporting import CellResult, Trial, aggregate

DEFAULT_RESULTS = Path("results/base-vs-tuned")
# The demo is deployed from `space/` alone, so its Docker build context cannot reach
# `results/`. Write a second copy there rather than duplicating the figure by hand and
# letting the deployed version drift from the committed one.
DEMO_ASSETS = Path("space/assets")

# Reference points the curve has to be read against. A tuned model that merely matches the
# prompted frontier ceiling would prove nothing, which is why the bar sits above it.
PROMPT_CEILING_ADHERENCE = 0.71
PROMPT_CEILING_ROBUSTNESS = 0.67
RELIABILITY_BAR = 0.80

# Two categorical slots from a validated palette, one per metric, in fixed order. Both
# modes clear the lightness band, chroma floor, colorblind separation, normal-vision
# floor, and 3:1 contrast against their own surface.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e2e1dd",
        "adherence": "#2a78d6",
        "robustness": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#33322f",
        "adherence": "#3987e5",
        "robustness": "#d95926",
    },
}


def _tick_int(value: float, _pos: float) -> str:
    """Format an axis tick as a bare integer."""
    return f"{int(value)}"


def _tick_percent(value: float, _pos: float) -> str:
    """Format an axis tick as a whole percentage."""
    return f"{int(value)}%"


def load_curve(results_dir: Path) -> list[CellResult]:
    """Read the curve points from an eval run.

    Args:
        results_dir: Directory holding `trials.jsonl`.

    Returns:
        Cells carrying a dataset size, ascending. The untuned base is excluded: it has no
        N, and plotting it at an implied zero would read as a trained checkpoint.

    Raises:
        SystemExit: If the eval has not been run, or produced no curve points.
    """
    path = results_dir / "trials.jsonl"
    if not path.exists():
        raise SystemExit(f"no eval results at {path}; run eval.py first")
    trials = [
        Trial.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    points = sorted(
        (c for c in aggregate(trials) if c.dataset_size is not None),
        key=lambda c: c.dataset_size or 0,
    )
    if not points:
        raise SystemExit("no trials carry a dataset_size; nothing to plot")
    return points


def render(points: list[CellResult], theme: dict[str, str], out_path: Path) -> None:
    """Draw the performance-vs-N curve.

    Args:
        points: Curve points, ascending by dataset size.
        theme: Surface, ink, and series colors for one mode.
        out_path: Where to write the PNG.
    """
    # Imported behind an explicit Any for the same reason as slm/sft.py: matplotlib's
    # partial type information produces dozens of "partially unknown" errors under pyright
    # strict, which bury real ones. Lazy so the eval path never pays for the import.
    matplotlib: Any = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt: Any = importlib.import_module("matplotlib.pyplot")
    ticker: Any = importlib.import_module("matplotlib.ticker")
    FuncFormatter, NullFormatter = ticker.FuncFormatter, ticker.NullFormatter

    sizes = [c.dataset_size or 0 for c in points]
    adherence = [c.spec_adherence * 100 for c in points]
    robustness = [c.robustness * 100 for c in points]

    fig: Any
    ax: Any
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    # References first, so the data sits on top of them. The two ceiling values are four
    # points apart, so they are drawn as one shaded band rather than two near-identical
    # dashed lines that would read as clutter and need two labels.
    ax.axhspan(
        PROMPT_CEILING_ROBUSTNESS * 100,
        PROMPT_CEILING_ADHERENCE * 100,
        color=theme["muted"],
        alpha=0.16,
        lw=0,
        zorder=1,
    )
    ax.annotate(
        "prompted frontier ceiling (67–71%)",
        xy=(sizes[-1], PROMPT_CEILING_ROBUSTNESS * 100),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=theme["muted"],
    )
    ax.axhline(RELIABILITY_BAR * 100, color=theme["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        "reliability bar (80%)",
        xy=(sizes[-1], RELIABILITY_BAR * 100),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=theme["muted"],
    )

    for values, key, label in (
        (adherence, "adherence", "Spec adherence (24 clean)"),
        (robustness, "robustness", "Robustness (12 adversarial)"),
    ):
        ax.plot(
            sizes, values, color=theme[key], lw=2, marker="o", markersize=8,
            # A surface-colored ring keeps the two markers legible where the series
            # converge at 100% and would otherwise overlap into one blob.
            markeredgecolor=theme["surface"], markeredgewidth=2, label=label, zorder=3,
        )
        # Label every point: only four per series, and the exact values are the finding.
        for x, y in zip(sizes, values):
            ax.annotate(
                f"{y:.0f}%", xy=(x, y), xytext=(0, 11), textcoords="offset points",
                ha="center", fontsize=8.5, color=theme["text"], zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(FuncFormatter(_tick_int))
    # A log axis labels its minor ticks by default ("6 x 10^1"), which collides with the
    # curve points those ticks sit between. The four N values are the only meaningful
    # positions on this axis.
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(sizes[0] * 0.82, sizes[-1] * 1.22)
    ax.set_ylim(-4, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.get_yaxis().set_major_formatter(FuncFormatter(_tick_percent))

    ax.set_xlabel("Training examples (N, log scale)", fontsize=10, color=theme["muted"])
    ax.set_title(
        "Data efficiency — Qwen3-0.6B, Python state-lifetime tutor",
        fontsize=12.5, color=theme["text"], pad=32, loc="left",
    )
    ax.annotate(
        "Untuned base scores 0% on both metrics with the full behavior spec as its prompt.",
        xy=(0, 1.045), xycoords="axes fraction", fontsize=9, color=theme["muted"],
    )

    ax.grid(axis="y", color=theme["grid"], lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)

    legend = ax.legend(
        loc="lower right", frameon=False, fontsize=9.5, handlelength=2.4, borderpad=0.3
    )
    for text in legend.get_texts():
        text.set_color(theme["text"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Render the data-efficiency curve in light and dark modes."""
    parser = argparse.ArgumentParser(description="Plot the data-efficiency curve")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--demo-assets", type=Path, default=DEMO_ASSETS)
    args = parser.parse_args()

    points = load_curve(args.results)
    for mode, theme in THEMES.items():
        render(points, theme, args.out / f"curve-{mode}.png")
        render(points, theme, args.demo_assets / f"curve-{mode}.png")

    summary: list[dict[str, Any]] = [
        {"n": c.dataset_size, "adherence": c.spec_adherence, "robustness": c.robustness}
        for c in points
    ]
    print(f"points: {summary}")


if __name__ == "__main__":
    main()

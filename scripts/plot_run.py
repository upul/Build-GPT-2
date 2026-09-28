#!/usr/bin/env python3
"""Plot a Weights & Biases training run as SVGs for a README.

Produces a light and a dark variant of each chart so they can be paired in a
``<picture>`` element. Charts whose metrics are absent from the run are skipped,
so the same script serves runs that log different things.

    uv run python scripts/plot_run.py <run-id> --entity upul --project build-gpt2

The metric names below default to this repository's logging scheme; pass the
flags to point the script at a differently instrumented run.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Validated categorical slots 1 and 2, stepped separately for each surface.
# See the dataviz reference palette; both modes pass all six checks.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e3e0",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#2f2f2d",
        "series": ["#3987e5", "#d95926"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a W&B run as README-ready SVGs")
    parser.add_argument("run_id", help="W&B run id (the short hash, not the display name)")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument(
        "--x",
        choices=["tokens", "step"],
        default="tokens",
        help="x-axis; tokens keeps runs at different batch sizes comparable",
    )
    parser.add_argument("--train-loss", default="train/loss")
    parser.add_argument("--val-loss", default="val/loss")
    parser.add_argument("--tokens", default="train/tokens")
    parser.add_argument("--eval-metric", default="hellaswag/acc_norm")
    parser.add_argument("--eval-label", default="HellaSwag acc_norm")
    parser.add_argument(
        "--eval-reference",
        type=float,
        default=None,
        help="draw a horizontal reference line, e.g. 29.55 for GPT-2 124M",
    )
    parser.add_argument("--eval-reference-label", default="reference")
    parser.add_argument("--throughput", default="train/token_per_sec")
    return parser.parse_args()


def fetch_history(entity: str, project: str, run_id: str, keys: list[str]):
    """Return a DataFrame of the run's full history for ``keys``."""
    import pandas as pd
    import wandb

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    wanted = {"_step", *keys}
    rows = [{k: v for k, v in row.items() if k in wanted} for row in run.scan_history()]
    return pd.DataFrame(rows), run


def _si_formatter(vmax: float) -> FuncFormatter:
    """Label token counts as 0.5B / 1B / 1.5B rather than matplotlib's 1e9 offset.

    One scale is chosen for the whole axis from its maximum, so ticks never mix
    units (a 500M label sitting next to a 1B label reads as a different axis).
    """
    scale, suffix = 1.0, ""
    for candidate, label in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if vmax >= candidate:
            scale, suffix = candidate, label
            break
    return FuncFormatter(lambda value, _pos: f"{value / scale:g}{suffix}")


def style_axes(ax, theme: dict, xlabel: str, ylabel: str, title: str, si_x: bool = False) -> None:
    ax.set_facecolor(theme["surface"])
    ax.figure.set_facecolor(theme["surface"])
    ax.set_title(title, color=theme["text"], fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=theme["muted"], fontsize=9)
    ax.set_ylabel(ylabel, color=theme["muted"], fontsize=9)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if si_x:
        ax.xaxis.set_major_formatter(_si_formatter(max(abs(v) for v in ax.get_xlim())))
    # Room on the right for the end-of-series labels.
    ax.margins(x=0.06)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(theme["grid"])


def series_xy(df, x_key: str, y_key: str):
    """Drop rows where either axis is missing; W&B pads absent metrics with NaN."""
    if y_key not in df.columns or x_key not in df.columns:
        return None, None
    sub = df[[x_key, y_key]].dropna().sort_values(x_key)
    if sub.empty:
        return None, None
    return sub[x_key].to_numpy(), sub[y_key].to_numpy()


def save(fig, out: Path, name: str, mode: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{mode}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    return path


def plot_loss(df, args, x_key: str, x_label: str, mode: str, si_x: bool = False) -> Path | None:
    theme = THEMES[mode]
    tx, ty = series_xy(df, x_key, args.train_loss)
    if tx is None:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(tx, ty, color=theme["series"][0], linewidth=2, label="train")
    ax.annotate(
        "train",
        (tx[-1], ty[-1]),
        color=theme["series"][0],
        fontsize=9,
        xytext=(6, -7),
        textcoords="offset points",
        va="center",
    )

    vx, vy = series_xy(df, x_key, args.val_loss)
    if vx is not None:
        ax.plot(
            vx, vy, color=theme["series"][1], linewidth=2, marker="o", markersize=4, label="val"
        )
        ax.annotate(
            "val",
            (vx[-1], vy[-1]),
            color=theme["series"][1],
            fontsize=9,
            xytext=(6, 7),
            textcoords="offset points",
            va="center",
        )
        # Two series: identity is never carried by colour alone.
        ax.legend(
            frameon=False,
            labelcolor=theme["muted"],
            fontsize=9,
            loc="upper right",
        )

    style_axes(ax, theme, x_label, "cross-entropy loss", "Training and validation loss", si_x)
    return save(fig, args.out, "loss", mode)


def plot_eval(df, args, x_key: str, x_label: str, mode: str, si_x: bool = False) -> Path | None:
    theme = THEMES[mode]
    ex, ey = series_xy(df, x_key, args.eval_metric)
    if ex is None:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if args.eval_reference is not None:
        ax.axhline(
            args.eval_reference,
            color=theme["muted"],
            linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
        ax.annotate(
            f"{args.eval_reference_label} · {args.eval_reference:g}%",
            (ex[0], args.eval_reference),
            color=theme["muted"],
            fontsize=9,
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.plot(ex, ey, color=theme["series"][0], linewidth=2, marker="o", markersize=5)
    ax.annotate(
        f"{ey[-1]:.2f}%",
        (ex[-1], ey[-1]),
        color=theme["series"][0],
        fontsize=9,
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
    )
    style_axes(ax, theme, x_label, args.eval_label, f"{args.eval_label} during training", si_x)
    return save(fig, args.out, "eval", mode)


def plot_throughput(
    df, args, x_key: str, x_label: str, mode: str, si_x: bool = False
) -> Path | None:
    theme = THEMES[mode]
    hx, hy = series_xy(df, x_key, args.throughput)
    if hx is None:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(hx, hy, color=theme["series"][0], linewidth=1.5)
    style_axes(ax, theme, x_label, "tokens / second", "Throughput", si_x)
    return save(fig, args.out, "throughput", mode)


def main() -> None:
    args = parse_args()
    keys = [args.train_loss, args.val_loss, args.eval_metric, args.throughput, args.tokens]
    df, run = fetch_history(args.entity, args.project, args.run_id, keys)
    print(f"{run.name} ({args.run_id}): {len(df)} history rows")

    if args.x == "tokens" and args.tokens in df.columns:
        x_key, x_label, si_x = args.tokens, "tokens seen", True
    else:
        x_key, x_label, si_x = "_step", "optimizer step", False
        if args.x == "tokens":
            print(f"note: {args.tokens!r} not logged, falling back to step")

    written = []
    for mode in ("light", "dark"):
        for fn in (plot_loss, plot_eval, plot_throughput):
            path = fn(df, args, x_key, x_label, mode, si_x)
            if path is not None:
                written.append(path)

    if not written:
        raise SystemExit("no plottable metrics found — check the --*-loss/--eval-metric names")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

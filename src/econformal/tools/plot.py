"""Plotting utilities for Econformal — conformal and traditional confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_ci_plot(
    plot_data: pd.DataFrame,
    time_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    coverage: float,
    treat_time,
    *,
    traditional: bool = False,
    trad_lower_col: str | None = None,
    trad_upper_col: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
) -> Figure:
    """Draw a conformal-inference confidence-interval plot.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Must contain columns ``time_col``, ``"effect"``, ``ci_lower_col``,
        ``ci_upper_col``, and when ``traditional=True`` also ``trad_lower_col``
        / ``trad_upper_col``.
    time_col : str
        Name of the time column (x-axis).
    ci_lower_col, ci_upper_col : str
        Column names for conformal lower / upper bounds.
    coverage : float
        Coverage level in (0, 1), e.g. 0.9 for 90%.
    treat_time : scalar
        x-coordinate for the vertical treatment reference line.
    traditional : bool
        If True, overlay traditional CI as I-bar error bars (工字型区间).
    trad_lower_col, trad_upper_col : str or None
        Column names for traditional lower / upper bounds (required when
        ``traditional=True``).
    figsize : tuple[float, float]
        (width, height) in inches; used only when *ax* is None.
    ax : Axes or None
        If provided, draw on this axes (enables subplot composition).
        If None, a new figure + axes is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    # ------ 1. copy to avoid mutating caller's DataFrame ------
    df = plot_data.copy()

    # ------ 2. set index on the copy, extract time values ------
    df = df.set_index(time_col)
    time_values = df.index.values
    x_min, x_max = time_values.min(), time_values.max()

    # ------ 3. axes setup ------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        if fig is None:
            raise ValueError("The provided *ax* is not attached to a figure.")

    # ------ 4. y-range ------
    mask = df[ci_lower_col].notna() & df[ci_upper_col].notna()
    y_min = df["effect"].min()
    y_max = df["effect"].max()
    if mask.any():
        y_min = min(y_min, df.loc[mask, ci_lower_col].min())
        y_max = max(y_max, df.loc[mask, ci_upper_col].max())
    pad = (y_max - y_min) * 0.05 or 0.5
    y_min, y_max = y_min - pad, y_max + pad

    # ------ 5. reference lines ------
    ax.axhline(y=0, color="black", linewidth=1.5)
    ax.axvline(x=treat_time, color="black", linestyle="--",
               linewidth=1.5, label="Treat")

    # ------ 6. effect line ------
    ax.plot(time_values, df["effect"], color="C1", linewidth=1.5,
            label="Effect")

    # ------ 7. conformal shaded area (浅蓝色) ------
    conformal_mask = df[ci_lower_col].notna() & df[ci_upper_col].notna()
    if conformal_mask.any():
        ax.fill_between(
            time_values[conformal_mask],
            df.loc[conformal_mask, ci_lower_col],
            df.loc[conformal_mask, ci_upper_col],
            alpha=0.2, color="lightskyblue",
            label=f"Conformal CI ({int(coverage * 100)}%)",
        )

    # ------ 8. traditional CI (工字型 I-bars) ------
    if traditional:
        if trad_lower_col is None or trad_upper_col is None:
            raise ValueError(
                "traditional=True requires trad_lower_col and trad_upper_col."
            )
        cap_width = (x_max - x_min) * 0.015
        if cap_width <= 0:
            cap_width = 0.2  # fallback for single time-point

        first_bar = True
        for idx, row in df.iterrows():
            low = row[trad_lower_col]
            high = row[trad_upper_col]
            if pd.isna(low) or pd.isna(high):
                continue
            lbl = f"Traditional CI ({int(coverage * 100)}%)" if first_bar else None
            # vertical stem
            ax.plot([idx, idx], [low, high], color="gray",
                    linewidth=1.5, alpha=0.8, label=lbl)
            # top cap
            ax.plot([idx - cap_width, idx + cap_width], [high, high],
                    color="gray", linewidth=1.5, alpha=0.8)
            # bottom cap
            ax.plot([idx - cap_width, idx + cap_width], [low, low],
                    color="gray", linewidth=1.5, alpha=0.8)
            first_bar = False

    # ------ 9. legend ------
    ax.legend(title=f"Coverage: {int(coverage * 100)}%")

    # ------ 10. axis limits & labels ------
    x_pad = (x_max - x_min) * 0.05 or 1.0
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(time_col)
    ax.set_ylabel("Effect")

    return fig

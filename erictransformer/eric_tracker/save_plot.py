from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from erictransformer.exceptions.eric_exceptions import EricPlotError

ERIC_RED = "#d62828"
ERIC_BLUE = "#1d84e2"
MID_GREY = "#4c4c4c"
BG_LIGHT = "#fafafa"


@dataclass
class StyleConfig:
    dpi: int = 150
    fig_size: Tuple[float, float] = (7.5, 4.5)

    colour_red: str = ERIC_RED
    colour_blue: str = ERIC_BLUE
    colour_light_grey: str = BG_LIGHT
    colour_grey: str = MID_GREY

    line_width: float = 2.5
    fill_alpha: float = 0.07
    line_shadow_alpha: float = 0.25
    line_shadow_offset: Tuple[float, float] = (-0.5, -0.5)

    marker_shape: str = "o"
    marker_size: int = 3
    marker_facecolor: str = "white"
    marker_edgecolor: str = "white"
    marker_edgewidth: float = 1.0
    marker_zorder: int = 1

    extrema_size: int = 120
    extrema_up_marker: str = "^"
    extrema_down_marker: str = "v"
    extrema_edgewidth: float = 1.5

    summary_xy: Tuple[float, float] = (0.50, 0.98)
    summary_fontsize: int = 8
    summary_box_kwargs: Dict[str, object] = field(
        default_factory=lambda: dict(
            boxstyle="round,pad=0.3",
            facecolor=BG_LIGHT,
            edgecolor=MID_GREY,
            alpha=0.85,
        )
    )

    grid_alpha: float = 0.15
    rc_extra: Dict[str, object] = field(default_factory=dict)

    def rc_params(self) -> Dict[str, object]:
        base = {
            "figure.dpi": self.dpi,
            "figure.facecolor": self.colour_light_grey,
            "axes.facecolor": self.colour_light_grey,
            "axes.edgecolor": "none",
            "grid.alpha": self.grid_alpha,
            "axes.labelcolor": self.colour_grey,
            "text.color": self.colour_grey,
            "xtick.color": self.colour_grey,
            "ytick.color": self.colour_grey,
        }
        base.update(self.rc_extra)
        return base

    def apply(self) -> None:
        mpl.rcParams.update(self.rc_params())

    def shadow(self) -> Sequence[pe.AbstractPathEffect]:
        return [
            pe.SimpleLineShadow(
                alpha=self.line_shadow_alpha, rho=self.line_shadow_offset
            ),
            pe.Normal(),
        ]


def save_train_eval_loss_plot(
    tracker_state_history,
    eval_steps_history,
    out_dir,
    style: StyleConfig = StyleConfig(),
):
    try:
        if not tracker_state_history:
            return

        style.apply()

        steps = np.array([s.current_step for s in tracker_state_history])
        train_loss = np.array([s.train_loss for s in tracker_state_history])
        eval_loss = np.array([s.eval_loss for s in tracker_state_history])

        eval_mask = np.isin(steps, np.array(eval_steps_history))
        eval_steps_plot = steps[eval_mask]
        eval_loss_plot = eval_loss[eval_mask]
        has_eval = eval_steps_plot.size > 0

        fig, ax = plt.subplots(figsize=style.fig_size)

        ax.fill_between(steps, train_loss, alpha=style.fill_alpha)
        ax.plot(
            steps,
            train_loss,
            label="train",
            linewidth=style.line_width,
            marker=style.marker_shape,
            markersize=style.marker_size,
            markerfacecolor=style.marker_facecolor,
            markeredgecolor=style.marker_edgecolor,
            markeredgewidth=style.marker_edgewidth,
            path_effects=style.shadow(),
            color=style.colour_red,
        )

        # eval curve
        if has_eval:
            ax.fill_between(
                eval_steps_plot,
                eval_loss_plot,
                alpha=style.fill_alpha,
                color=style.colour_blue,
            )
            ax.plot(
                eval_steps_plot,
                eval_loss_plot,
                label="eval",
                linewidth=style.line_width,
                marker=style.marker_shape,
                markersize=style.marker_size,
                markerfacecolor=style.marker_facecolor,
                markeredgecolor=style.marker_edgecolor,
                markeredgewidth=style.marker_edgewidth,
                path_effects=style.shadow(),
                color=style.colour_blue,
            )

        # extrema markers
        def mark(idx, xs, ys, marker, color):
            ax.scatter(
                xs[idx],
                ys[idx],
                s=style.extrema_size,
                marker=marker,
                zorder=style.marker_zorder,
                color=color,
                edgecolor=style.marker_edgecolor,
                linewidth=style.extrema_edgewidth,
            )

        mark(
            np.argmin(train_loss),
            steps,
            train_loss,
            style.extrema_down_marker,
            style.colour_red,
        )
        mark(
            np.argmax(train_loss),
            steps,
            train_loss,
            style.extrema_up_marker,
            style.colour_red,
        )
        if has_eval:
            mark(
                np.argmin(eval_loss_plot),
                eval_steps_plot,
                eval_loss_plot,
                style.extrema_down_marker,
                style.colour_blue,
            )
            mark(
                np.argmax(eval_loss_plot),
                eval_steps_plot,
                eval_loss_plot,
                style.extrema_up_marker,
                style.colour_blue,
            )

        # summary box
        summary = (
            f"max train: {train_loss.max():.3f}\nmin train: {train_loss.min():.3f}\n"
        )
        if has_eval:
            summary += (
                f"max eval : {eval_loss_plot.max():.3f}\n"
                f"min eval : {eval_loss_plot.min():.3f}\n"
            )

        ax.text(
            *style.summary_xy,
            summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=style.summary_fontsize,
            color=style.colour_grey,
            bbox=style.summary_box_kwargs,
        )

        ax.set_xlabel("Step", labelpad=6)
        ax.set_ylabel("Loss", labelpad=6)
        ax.grid(True)
        ax.legend(frameon=False, loc="upper right")
        ax.set_xlim(left=steps.min())  # start at step 0
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # show only whole‑number ticks
        fig.tight_layout()

        # write file
        plot_path = Path(out_dir) / "loss_curve.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=style.dpi)
        plt.close(fig)
    except Exception as e:
        raise EricPlotError(f"Error saving train/eval loss plot: {e}")


def save_lr_plot(
    tracker_state_history,
    out_dir,
    style: StyleConfig = StyleConfig(),
):
    try:
        if not tracker_state_history:
            return

        style.apply()

        steps = np.array([s.current_step for s in tracker_state_history])
        lr = np.array([s.lr for s in tracker_state_history])

        fig, ax = plt.subplots(figsize=style.fig_size)

        ax.fill_between(steps, lr, alpha=style.fill_alpha)
        ax.plot(
            steps,
            lr,
            label="learning‑rate",
            linewidth=style.line_width,
            marker=style.marker_shape,
            markersize=style.marker_size,
            markerfacecolor=style.marker_facecolor,
            markeredgecolor=style.marker_edgecolor,
            markeredgewidth=style.marker_edgewidth,
            path_effects=style.shadow(),
            color=style.colour_red,
        )

        # extrema markers
        def mark(idx, xs, ys, marker, color):
            ax.scatter(
                xs[idx],
                ys[idx],
                s=style.extrema_size,
                marker=marker,
                zorder=style.marker_zorder,
                color=color,
                edgecolor=style.marker_edgecolor,
                linewidth=style.extrema_edgewidth,
            )

        mark(np.argmin(lr), steps, lr, style.extrema_down_marker, style.colour_red)
        mark(np.argmax(lr), steps, lr, style.extrema_up_marker, style.colour_red)

        # summary box
        summary = f"max lr : {lr.max():.6f}\nmin lr : {lr.min():.6f}\n"

        ax.text(
            *style.summary_xy,
            summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=style.summary_fontsize,
            color=style.colour_grey,
            bbox=style.summary_box_kwargs,
        )

        ax.set_xlabel("Step", labelpad=6)
        ax.set_ylabel("Learning Rate", labelpad=6)
        ax.grid(True)
        ax.legend(frameon=False, loc="upper right")
        ax.set_xlim(left=steps.min())  # start at step 0
        ax.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # show only whole‑number ticks

        fig.tight_layout()

        # write file
        plot_path = Path(out_dir) / "lr_curve.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=style.dpi)
        plt.close(fig)

    except Exception as e:
        raise EricPlotError(f"Error saving lr plot: {e}")


def save_metric_plots(
    tracker_state_history,
    out_dir,
    style: StyleConfig = StyleConfig(),
):
    try:
        if not tracker_state_history:
            return

        style.apply()

        # initialize an empty list of metrics
        all_metrics = {}
        for step_idx, state in enumerate(tracker_state_history):
            for metric_name, value in state.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(
                    (state.current_step, value)
                )  # Track step + value

        # plot each metric separately
        for metric_name, data in all_metrics.items():
            steps, values = zip(*data)

            # history → arrays
            steps = np.array(steps)
            values = np.array(values)

            # plotting
            fig, ax = plt.subplots(figsize=style.fig_size)

            ax.fill_between(steps, values, alpha=style.fill_alpha)
            ax.plot(
                steps,
                values,
                label=metric_name,
                linewidth=style.line_width,
                marker=style.marker_shape,
                markersize=style.marker_size,
                markerfacecolor=style.marker_facecolor,
                markeredgecolor=style.marker_edgecolor,
                markeredgewidth=style.marker_edgewidth,
                path_effects=style.shadow(),
                color=style.colour_red,
            )

            # extrema markers
            def mark(idx, xs, ys, marker, color):
                ax.scatter(
                    xs[idx],
                    ys[idx],
                    s=style.extrema_size,
                    marker=marker,
                    zorder=style.marker_zorder,
                    color=color,
                    edgecolor=style.marker_edgecolor,
                    linewidth=style.extrema_edgewidth,
                )

            mark(
                np.argmin(values),
                steps,
                values,
                style.extrema_down_marker,
                style.colour_red,
            )
            mark(
                np.argmax(values),
                steps,
                values,
                style.extrema_up_marker,
                style.colour_red,
            )

            # summary box
            summary = (
                f"max {metric_name} : {values.max():.6f}\n"
                f"min {metric_name} : {values.min():.6f}\n"
            )

            ax.text(
                *style.summary_xy,
                summary,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=style.summary_fontsize,
                color=style.colour_grey,
                bbox=style.summary_box_kwargs,
            )

            ax.set_xlabel("Step", labelpad=6)
            ax.set_ylabel(f"{metric_name}", labelpad=6)
            ax.grid(True)
            ax.legend(frameon=False, loc="upper right")
            ax.set_xlim(left=steps.min())  # start at step 0
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # show only whole‑number ticks

            fig.tight_layout()

            # write file
            plot_path = Path(out_dir) / f"{metric_name}.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=style.dpi)
            plt.close(fig)

    except Exception as e:
        raise EricPlotError(f"Error saving lr plot: {e}")

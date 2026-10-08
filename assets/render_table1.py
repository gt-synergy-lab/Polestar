"""Render the published 256-token accuracy and throughput comparison."""

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Bbox


METHODS = ["Baseline", "Baseline+Parallel", "Fast-dLLM", "Dynamic-dLLM", "Elastic-Cache", "Polestar"]
MODELS = ["LLaDA-8B-Instruct", "Dream-7B-Instruct"]
BENCHMARKS = ["GSM8K", "MATH", "HumanEval"]
ACCURACY_LIMITS = [100, 50, 70]
TPS_LIMITS = [120, 120, 160]
INK, GRID, POLESTAR = "#202020", "#DDDDDD", "#76B900"
THROUGHPUT = "#3279A8"
THROUGHPUT_LABEL = "#123F5D"
ACCURACY_LABEL_SIZE, TPS_LABEL_SIZE = 12, 10.5


def place_labels(fig, panels):
    """Keep numbers clear of curves, markers, bar tops, and other numbers."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    margin = 1.5 * fig.dpi / 100
    moved = []
    placements = []
    for ax, throughput, bars, line, accuracy_labels, tps_labels, values in panels:
        points = throughput.transData.transform(line.get_xydata())
        segments = [PlotPath([a, b]) for a, b in zip(points[:-1], points[1:])]
        markers = []
        for i, (x, y) in enumerate(points):
            radius = (8 if i == 5 else 5) * fig.dpi / 72
            markers.append(Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius))
        bar_boxes = [bar.get_window_extent(renderer) for bar in bars]
        bar_tops = [PlotPath([(b.x0, b.y1), (b.x1, b.y1)]) for b in bar_boxes]
        for i, label in enumerate(accuracy_labels):
            for padding in (4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36):
                label.set_position((0, padding))
                box = label.get_window_extent(renderer).padded(margin)
                curve_box = box.padded(margin)
                other_boxes = [other.get_window_extent(renderer).padded(margin)
                               for other in accuracy_labels if other is not label]
                if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                        and not any(box.overlaps(b) for b in bar_boxes + markers + other_boxes)
                        and not any(path.intersects_bbox(curve_box, filled=False)
                                    for path in segments + bar_tops)):
                    break
            else:
                raise ValueError(f"No clear label position: {values[i]['model']}, {values[i]['benchmark']}, {values[i]['method']}")
            if padding > 8:
                lower_edge = ax.transData.inverted().transform((box.x0, box.y0))[1]
                ax.plot([i, i], [values[i]["accuracy"] + 0.5, lower_edge - 0.7],
                        color="#888888", linewidth=0.55, zorder=3)
                moved.append({"model": values[i]["model"], "benchmark": values[i]["benchmark"],
                              "method": values[i]["method"], "padding_points": padding})
        for i, label in enumerate(tps_labels):
            other_boxes = [other.get_window_extent(renderer).padded(margin)
                           for other in accuracy_labels + tps_labels[:i]]
            candidates = [(side, padding) for padding in (8, 10, 12, 14, 18, 22, 26, 30)
                          for side in ("above", "below")]
            for side, padding in candidates:
                label.set_position((0, padding if side == "above" else -padding))
                label.set_verticalalignment("bottom" if side == "above" else "top")
                raw_box = label.get_window_extent(renderer)
                box = raw_box.padded(margin)
                bar_box = bar_boxes[i]
                if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                        and bar_box.contains(raw_box.x0, raw_box.y0)
                        and bar_box.contains(raw_box.x1, raw_box.y1)
                        and not any(box.overlaps(b) for b in markers + other_boxes)
                        and not any(path.intersects_bbox(box.padded(margin), filled=False)
                                    for path in segments + bar_tops)):
                    break
            else:
                raise ValueError(f"No clear TPS label position: {values[i]['model']}, {values[i]['benchmark']}, {values[i]['method']}")
            placements.append({"model": values[i]["model"], "benchmark": values[i]["benchmark"],
                               "method": values[i]["method"], "side": side, "padding_points": padding})
        # Filled bars can carry TPS numbers; their height-defining tops stay clear.
        labels = accuracy_labels + tps_labels
        boxes = [label.get_window_extent(renderer).padded(margin) for label in labels]
        for i, box in enumerate(boxes):
            assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
            assert not any(box.overlaps(b) for b in markers + boxes[i + 1:])
            assert not any(path.intersects_bbox(box.padded(margin), filled=False)
                           for path in segments + bar_tops)
        for bar_box, label in zip(bar_boxes, tps_labels):
            box = label.get_window_extent(renderer)
            assert bar_box.contains(box.x0, box.y0) and bar_box.contains(box.x1, box.y1)
    return {"accuracy_labels": sum(len(p[4]) for p in panels),
            "tps_labels": sum(len(p[5]) for p in panels), "overlaps": 0,
            "labels_outside_axes": 0, "tps_labels_outside_bars": 0,
            "moved_accuracy_labels": moved, "tps_placements": placements}


def render(rows, output):
    selected = {(r["model"], r["benchmark"], r["method"]): r
                for r in rows if r["generation_length"] == 256}
    assert len(selected) == 36, "Expected six methods in each of six panels"
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 14,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
        "svg.fonttype": "none", "svg.hashsalt": "polestar-table1",
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=200, facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.945, top=0.85, bottom=0.175,
                        wspace=0.48, hspace=1.05)
    legend = [
        Patch(facecolor="#CCCCCC", edgecolor="#555555", label="Accuracy (%)"),
        Line2D([], [], color=THROUGHPUT, marker="s", linewidth=2,
               markersize=6, label="TPS"),
        (Patch(facecolor=POLESTAR, edgecolor="#446A00"),
         Line2D([], [], color=THROUGHPUT, marker="*", linestyle="none", markersize=11)),
    ]
    fig.legend(handles=legend, labels=["Accuracy (%)", "TPS", "Polestar"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
               loc="upper center", bbox_to_anchor=(0.515, 0.985),
               ncol=3, frameon=False, prop={"size": 15, "weight": "bold"},
               handlelength=1.8, columnspacing=2.2)

    panels = []
    for row_index, model in enumerate(MODELS):
        for column, benchmark in enumerate(BENCHMARKS):
            ax = axes[row_index, column]
            throughput = ax.twinx()
            values = [selected[(model, benchmark, method)] for method in METHODS]
            x = list(range(len(METHODS)))
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=GRID, linewidth=0.65)
            bars = ax.bar(x, [v["accuracy"] for v in values], width=0.64,
                          color=["#CCCCCC"] * 5 + [POLESTAR], edgecolor="#555555",
                          linewidth=0.75, zorder=2)
            line, = throughput.plot(x, [v["tps"] for v in values], color=THROUGHPUT,
                                    marker="s", markevery=list(range(5)),
                                    linewidth=2, markersize=5.5, zorder=3)
            pole = values[-1]
            throughput.scatter(5, pole["tps"], s=150, marker="*", facecolors=THROUGHPUT,
                               edgecolors=THROUGHPUT_LABEL, linewidths=0.8, zorder=4)
            ax.set_ylim(0, ACCURACY_LIMITS[column])
            ax.set_yticks(([0, 25, 50, 75, 100], [0, 10, 20, 30, 40, 50], [0, 20, 40, 60])[column])
            throughput.set_ylim(0, TPS_LIMITS[column])
            throughput.set_yticks([0, TPS_LIMITS[column] / 4, TPS_LIMITS[column] / 2,
                                  TPS_LIMITS[column] * 3 / 4, TPS_LIMITS[column]])
            ax.set_xlim(-0.6, 5.6)
            ax.set_xticks(x, METHODS, fontsize=14, rotation=45, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", length=0, pad=6)
            ax.get_xticklabels()[-1].set_color("#446A00")
            ax.get_xticklabels()[-1].set_weight("bold")
            ax.set_xlabel("Method", fontsize=14, weight="bold", labelpad=7)
            ax.set_ylabel("Accuracy (%)", fontsize=15, weight="bold", labelpad=5)
            throughput.set_ylabel("TPS (tokens/s)", fontsize=15, weight="bold", labelpad=5)
            labels = []
            tps_labels = []
            for i, value in enumerate(values):
                ours = i == 5
                accuracy_label = str(Decimal(str(value["accuracy"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
                labels.append(ax.annotate(accuracy_label, (i, value["accuracy"]),
                                         xytext=(0, 4), textcoords="offset points", ha="center",
                                         fontsize=ACCURACY_LABEL_SIZE, color="#446A00" if ours else INK,
                                         weight="bold" if ours else "normal", zorder=6))
                tps_label = str(Decimal(str(value["tps"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
                tps_labels.append(throughput.annotate(tps_label, (i, value["tps"]),
                                  xytext=(0, 8), textcoords="offset points", ha="center",
                                  va="bottom", fontsize=TPS_LABEL_SIZE, color=THROUGHPUT_LABEL,
                                  zorder=6))
            panels.append((ax, throughput, bars, line, labels, tps_labels, values))
            if row_index == 0:
                ax.set_title(benchmark, fontsize=18, weight="bold", pad=13)
            for target in (ax, throughput):
                target.spines["top"].set_visible(False)
                target.tick_params(axis="y", labelsize=12, pad=3, length=3)
            throughput.spines["left"].set_visible(False)
            throughput.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
        position = axes[row_index, 0].get_position()
        fig.text(0.012, (position.y0 + position.y1) / 2, model, rotation=90,
                 fontsize=15, weight="bold", ha="center", va="center")
    layout = place_labels(fig, panels)
    output.mkdir(exist_ok=True)
    stem = output / "table1-256"
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.03, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.03,
                metadata={"Source": "https://arxiv.org/pdf/2607.14107, Table 1, generation length 256"})
    plt.close(fig)
    return layout


def main():
    assets = Path(__file__).resolve().parent
    data = json.loads((assets / "data/table1.json").read_text())
    print(json.dumps(render(data["rows"], assets / "figures"), indent=2))


if __name__ == "__main__":
    main()

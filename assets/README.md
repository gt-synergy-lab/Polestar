# Figures

| Asset | Source |
|---|---|
| `figures/accuracy-throughput.png` | Figure 1 of [Polestar, arXiv:2607.14107](https://arxiv.org/pdf/2607.14107) |
| `figures/polestar-overview.png` | Figure 5 of the same paper |
| `figures/table1-256.png` / `.svg` | Table 1 of the same paper, generation length 256 |

Figure 1 and Figure 5 are exports of the original paper figures with their visual content preserved. README and documentation share the same files.

The comparison figure plots Table 1 at generation length 256: six methods, two model rows, and three benchmark columns. Bars show accuracy (%) on the left axis; the blue line shows TPS on the right axis. Both axes start at zero, with scales shared between models within each benchmark column. Every accuracy bar and TPS point is annotated to one decimal place; the source data retains full precision. Accuracy labels use 12pt type; TPS labels use 10.5pt dark-blue type and stay inside their corresponding bars. TPS labels are placed above or below each point to avoid curves, markers, bar tops, other labels, and the plot boundary. Polestar uses NVIDIA green (`#76B900`) for its bar and a blue star for its TPS point. HumanEval accuracy is pass@1.

Method labels use full names: Baseline, Baseline+Parallel, Fast-dLLM, Dynamic-dLLM, Elastic-Cache, and Polestar. Rows use LLaDA-8B-Instruct and Dream-7B-Instruct. GSM8K is 5-shot, MATH is 4-shot, and HumanEval is 0-shot. Measurements use one NVIDIA A100 80GB and batch size 1; see the [results guide](../docs/results.md) for metric definitions.

The complete Table 1 values are stored in `data/table1.json`; the renderer selects the 36 records at generation length 256. To regenerate the PNG and SVG files:

```bash
python -m pip install matplotlib==3.9.0
python assets/render_table1.py
```

Keep aspect ratios intact. Figure 1 occupies a separate README row at 480px. The comparison grid and method overview use the full reading width. Readers can open either performance figure at full size, including a vector SVG for the comparison grid.

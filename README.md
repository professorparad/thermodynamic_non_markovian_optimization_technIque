# Thermodynamic Non-Markovian Optimization Framework

## Structure
- `main.py`: Repo-root CLI entry point.
- `phase1/project_root/src/`: Core landscapes, optimizers, device helpers, memory kernels, and MPS/topology utilities.
- `phase1/project_root/benchmarks/`: Benchmark sweeps, including the Hybrid v4.2 adaptive benchmark from the pasted script.
- `phase1/project_root/plots/`: Plotting helpers.
- `outputs/`: Generated plots and CSV reports.

## Install
Install the scientific Python dependencies:

```bash
pip install -r requirements.txt
```

For CUDA acceleration, install a CUDA-enabled PyTorch build that matches your GPU and driver. Then run with `--device cuda` to require CUDA, or `--device auto` to use CUDA when available and CPU otherwise.

## Usage
Run a quick optimizer comparison:

```bash
python main.py --sweep compare --landscape rastrigin --dim 10 --steps 200 --starts 8 --device auto
```

The comparison now includes `Global Hybrid`, a multi-start scout/refine optimizer intended for finding better global minima on multimodal landscapes than a single Adam run.

Run every landscape comparison from one terminal command and save visual outputs:

```bash
python main.py --sweep landscapes --dim 10 --steps 500 --trials 3 --starts 16 --device cuda --save outputs/all_landscapes
```

This writes per-landscape loss curves, `all_landscape_summary.png`, `error_vs_reconstruction.png`, histories CSVs, and `all_landscape_comparison.csv`.

Run the pasted Hybrid v4.2 adaptive benchmark:

```bash
python main.py --sweep v42 --steps 800 --lr 0.01 --device cuda --save outputs/v4_2
```

Use CPU explicitly:

```bash
python main.py --sweep v42 --device cpu
```

Generated files:

- `hybrid_v4_2_vs_adam.csv`
- `hybrid_v4_2_vs_adam.png`

Compatibility wrapper:

```bash
python phase1/project_root/main.py
```

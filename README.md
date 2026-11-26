# FAST-Brain

*FAST-Brain: A Flow-Aligned Spatio-Temporal Surrogate Brain Model*

Shucheng Liu, Chengchun Shi, Kai Zhang, and Hongtu Zhu · NeurIPS 2026. [Paper](https://arxiv.org/abs/2609.34354)

FAST-Brain forecasts regional BOLD signals from a window of observed brain activity. It combines a temporal Transformer with graph convolutions over spatial brain priors and generates future trajectories using flow matching.

## Installation

Use Python 3.10 or later. The training script uses CUDA when available.

```bash
git clone https://github.com/shucheng2003/FAST-Brain.git
cd FAST-Brain
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run the simulations

```bash
python -m fast_brain.run --dataset rnn
python -m fast_brain.run --dataset sddes
```

| Dataset | Regions | Time points | Training epochs | Seed |
| --- | ---: | ---: | ---: | ---: |
| RNN | 20 | 8,000 | 150 | 7 |
| SDDEs | 66 | 3,000 | 180 | 37 |

Each run trains one model and evaluates ten generations with the fitted weights. The dataset configurations specify the model, training, and evaluation parameters. [RNN configuration](configs/rnn.yaml) · [SDDEs configuration](configs/sddes.yaml)

## Outputs

Each run writes `metrics.jsonl` and `summary.csv` under `results/<dataset>/fast/`. Add `--save-checkpoint` to save the fitted weights, or pass an absolute path to `--results-dir` to choose an output directory.

Both simulations report functional connectivity (FC) correlation and mean absolute error. The reference FC is computed over the full observed signal, and both metrics use off-diagonal entries. On RNN, `ec_corr` and `ec_mae` compare the sparsified one-step Jacobian with the continuous reference. The classification metrics `ec_auroc` and `ec_auprc` use the input structural connectivity matrix thresholded at 0.05. RNN also reports CRPS, QICE, ProbCorr, and conditional FID.

## HCP data

The HCP resting-state fMRI example is available from the NPI repository accompanying Luo et al.'s *Nature Methods* paper. [Download the fMRI data](https://github.com/ncclab-sustech/NPI/tree/main/real_fMRI_data) · [Nature Methods paper](https://doi.org/10.1038/s41592-025-02654-x)

To run HCP from a source checkout, set the four session paths in `data.sessions` to your local REST1 and REST2 files in the listed LR/RL order. Set the paths in `data.matrices` to the corresponding `sc`, `length`, `network`, and `number` matrices in the same region order. You can use absolute paths or paths relative to the repository root. [HCP configuration](configs/hcp.yaml)

```bash
python -m fast_brain.run --dataset hcp
```

## Acknowledgments

The RNN simulation follows Luo et al. The SDDEs simulation uses the framework described by Sanz Leon et al. The conditional FID evaluator uses TS2Vec by Yue et al. [NPI](https://github.com/ncclab-sustech/NPI) · [The Virtual Brain](https://doi.org/10.3389/fninf.2013.00010) · [TS2Vec](https://github.com/zhihanyue/ts2vec) · [Third-party licenses](third_party)

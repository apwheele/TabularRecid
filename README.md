# TabularRecid

Tabular foundation models for recidivism prediction with small samples, on the
NIJ Recidivism Forecasting Challenge data.

NVIDIA's Kumo Tabular is compared to logistic regression, a random forest,
LightGBM, and CatBoost. Each model is fit to ten random training samples of
500 to 16,000 people and to the full NIJ training sample, and scored on the
NIJ test sample, for all three rounds of the challenge. Rounds 2 and 3 are fit
only to people with no arrest in earlier years. The appendix adds TabICLv2 and
TabPFN v2, raw versus engineered features, and the NIJ false positive rates.

The paper is `paper.pdf` (also built as `paper.md`), from `paper.qmd` and
`references.bib`.

## Reproducing

This project uses [uv](https://docs.astral.sh/uv/) for the Python environment
and [Quarto](https://quarto.org/) to render the paper. The foundation models
need a CUDA GPU to run in reasonable time; the small Kumo model used about
1 GB of GPU memory.

```bash
uv sync
uv run python -m ipykernel install --user --name tabrecid --display-name "Python (tabrecid)"

# sample-size runs, raw features (resumable; results/samples/*.csv)
uv run python scripts/02_samples.py --models logit,rf,lgbm,catboost --features raw
uv run python scripts/02_samples.py --models kumo_small --features raw
# engineered features, full training sample only
uv run python scripts/02_samples.py --models logit,rf,lgbm,catboost,kumo_small --features fe --sizes full
# risk set versus training on everyone, rounds 2 and 3
uv run python scripts/03_conditional.py --models catboost,kumo_small
# appendix models
uv run python scripts/02_samples.py --models tabicl --features raw
uv run python scripts/02_samples.py --models tabpfn --features raw --max-size 2000 --reps 3

quarto render paper.qmd
```

`scripts/run_cpu_queue.sh` and `scripts/run_gpu_queue.sh` run the same steps as
two single-threaded queues. The results the paper reads are committed, so the
paper can be rendered without refitting anything. Run the tests with
`uv run pytest`.

## Layout

- `paper.qmd`, `references.bib` -- paper source and bibliography.
- `src/tabrecid/` -- data and features (`data.py`), model wrappers (`models.py`),
  metrics (`metrics.py`), and result summaries (`report.py`).
- `scripts/` -- the experiment runners.
- `data/` -- the public NIJ full dataset and the published leaderboard scores.
- `results/` -- per-run metrics, full-sample test predictions, and the risk set comparison.
- `filters/` -- pandoc Lua filter for the Markdown build.
- `tests/` -- pytest unit tests.

## Data

`data/NIJ_s_Recidivism_Challenge_Full_Dataset.csv` is the public NIJ Recidivism
Forecasting Challenge full dataset, originally distributed at
<https://data.ojp.usdoj.gov/Courts/NIJ-s-Recidivism-Challenge-Full-Dataset/ynf5-u8nk>.
`data/nij_leaderboard.csv` holds the top and last-listed scores from
<https://nij.ojp.gov/funding/recidivism-forecasting-challenge-results>.

## License

MIT, see `LICENSE`.

# Benchmark on a free Kaggle T4

A Kaggle notebook with a T4 is the cheapest way to measure the GPU of an AWS `g4dn.xlarge`: it is
free (within Kaggle's weekly GPU quota) and needs no cloud account. The notebook
[kaggle_t4_benchmark.ipynb](kaggle_t4_benchmark.ipynb) runs the same vision and audio benchmarks as
on the local machine, with `BENCH_ENV=kaggle-t4`, so the results can be compared side by side.

## Steps

1. On [kaggle.com](https://www.kaggle.com), *Create > New Notebook*, then *File > Import Notebook*
   and pick `kaggle_t4_benchmark.ipynb`.
2. *Session options*: Accelerator **GPU T4 x2**, Internet **on** (a phone-verified account is
   needed for both).
3. Run the cells in order (about 20 minutes in total).
4. Download `/kaggle/working/kaggle-t4.tar.gz`, then **Stop session**.
5. Import the records locally and compare:

```bash
scripts/import_kaggle.sh ~/Downloads/kaggle-t4.tar.gz
python -m src.runner.summarize vision --by-env --tag conc-c1 --tag conc-c2 --tag conc-c4
python -m src.runner.summarize audio --by-env --tag ls66-c1 --tag ls66-c2 --tag ls66-c4 --tag ls66-c8
python -m src.cost.report
```

## What runs on the T4

Everything is done by [scripts/kaggle_t4_bench.sh](../scripts/kaggle_t4_bench.sh): it checks the
GPU, the internet access and the checksums of the inputs, installs vLLM 0.29.0 and the client in
two separate environments, starts one model at a time with the same settings as Docker Compose
(plus `--dtype float16`, since the T4 has no bf16), samples the vLLM scheduler during the runs and
archives everything. Details in the [usage guide](../docs/usage.md#measurements-on-a-t4-kaggle).

## Good to know

- Only the first GPU is used: one model at a time, as on a single-GPU instance.
- Qwen3-VL needs `--mm-encoder-attn-backend TORCH_SDPA` on Turing (already set by the script).
- If a phase fails, the script skips it and goes on; `PHASES=audio` (or `vision`) replays one phase
  in the same session without losing the records already written.
- `%%bash` cells stop at the first error (`set -e`): never paste them into a local terminal.

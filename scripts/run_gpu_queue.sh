#!/bin/bash
# GPU jobs, run one after another. Each script skips work that is already saved.
# OMP_NUM_THREADS keeps the CPU side of the GPU jobs to one core.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TABRECID_THREADS=1

# wait for any Kumo raw run already in progress
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*scripts/0*kumo_small*' }" | grep -q CommandLine; do
  sleep 60
done

uv run python scripts/02_samples.py --models kumo_small --features raw >> logs/kumo_raw.log 2>&1
uv run python scripts/02_samples.py --models kumo_small --features fe --sizes full >> logs/kumo_fe.log 2>&1
uv run python scripts/03_conditional.py --models catboost,kumo_small >> logs/conditional.log 2>&1
uv run python scripts/02_samples.py --models tabicl --features raw >> logs/tabicl.log 2>&1
uv run python scripts/02_samples.py --models tabpfn --features raw --max-size 2000 --reps 3 >> logs/tabpfn.log 2>&1
echo "GPU QUEUE DONE" >> logs/gpu_queue.log

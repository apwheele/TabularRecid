#!/bin/bash
# CPU jobs, one thread, run one after another. Each script skips work already saved.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TABRECID_THREADS=1

while powershell -c "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*rf,lgbm,catboost*' }" | grep -q CommandLine; do
  sleep 60
done

uv run python scripts/02_samples.py --models rf,lgbm,catboost --features raw >> logs/trees_raw.log 2>&1
uv run python scripts/02_samples.py --models rf,lgbm,catboost --features fe --sizes full >> logs/trees_fe.log 2>&1
echo "CPU QUEUE DONE" >> logs/cpu_queue.log

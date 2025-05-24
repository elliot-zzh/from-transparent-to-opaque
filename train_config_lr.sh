#!/bin/bash

gpu_queue=($(seq 0 0))
pending_tasks=()
pending_tasks+=("--config configs/config_lr_0.toml
")pending_tasks+=("--config configs/config_lr_1.toml
")pending_tasks+=("--config configs/config_lr_2.toml
")
pids=()
launch_next_task() {
  if [ ${#pending_tasks[@]} -eq 0 ]; then return; fi
  if [ ${#gpu_queue[@]} -eq 0 ]; then return; fi
  gpu=${gpu_queue[0]}
  gpu_queue=("${gpu_queue[@]:1}")
  task=${pending_tasks[0]}
  pending_tasks=("${pending_tasks[@]:1}")
  accelerate launch --gpu_ids=$gpu --num_processes=1 --mixed_precision=bf16 train.py $task --traindataset dataset/train.jsonl --testdataset dataset/test.jsonl &
  pids+=($!)
}

while [ ${#pending_tasks[@]} -gt 0 ] || [ ${#pids[@]} -gt 0 ]; do
  while [ ${#pending_tasks[@]} -gt 0 ] && [ ${#gpu_queue[@]} -gt 0 ]; do launch_next_task; done
  for i in $(seq 0 $((${#pids[@]} - 1)) ); do
    if ! ps -p ${pids[$i]} > /dev/null; then
      gpu_used=$(grep -oP "gpu_ids=\K[0-9]+" <(ps -p ${pids[$i]} -o cmd=) 2>/dev/null || echo -1
      if [ $gpu_used -ne -1 ]; then gpu_queue+=($gpu_used); fi
      unset pids[$i]
      pids=("${pids[@]}")
    fi
  done
  sleep 1
done
wait

#!/bin/bash

./train_lr.sh
wait
best_lr=$(awk 'NR==1{print $1}' score_lr.txt 2>/dev/null)
if [ -z "$best_lr" ]; then
  best_lr=1e-06
fi
echo "Best lr: $best_lr"
python -c "import toml; config = toml.load('config.toml'); config['training']['lr'] = float(os.environ.get('best_lr', 1e-06)); toml.dump(config, 'final_best_config.toml')"
./train_other.sh
wait

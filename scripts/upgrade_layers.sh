#!/bin/zsh
# Re-convert the given backbone layers' routed experts at a new layout (format.expert_layout_for),
# in place, one layer at a time: drop the layer's progress entries and file, let the converter
# redo just those tasks. Usage: scripts/upgrade_layers.sh 21 22 23 ...
set -e
ROOT=$HOME/models/dsv41-mac
cd $HOME/DeepInfer-Mac && source .venv/bin/activate
for L in "$@"; do
  F=$(printf "L%02d" $L)
  echo "$(date '+%T') layer $L -> $(python -c "import sys; sys.path.insert(0,'.'); from deepinfer_mac import format as F; print(F.expert_layout_for($L))")"
  grep -v "^XL-$L-" $ROOT/progress.log > $ROOT/progress.tmp && mv $ROOT/progress.tmp $ROOT/progress.log
  rm -f $ROOT/experts/$F.bin
  python -u -m deepinfer_mac.convert.remote --out $ROOT --workers 6 --only "^XL-$L-" | tail -1
done
echo "$(date '+%T') upgrade done"

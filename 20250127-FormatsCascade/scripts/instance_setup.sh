# Manual steps:
#  scp ~/.gitconfig INSTANCE:~
#  mkdir nethome/USER
#  cd nethome/USER && git clone git@github.com:graphcore-research/squashed-llama.git --branch notebooks SquashedLlama
#  ln -s ~/nethome/USER/SquashedLlama/20250127-FormatsCascade/ work

set -e
set -o xtrace

mkdir -p ~/block-formats/out
ln -s ~/block-formats/out out || true

aws configure

aws s3 sync s3://graphcore-research/2025-04-block-formats/20250423-fisher/ out/20250423-fisher/

python3 -m venv ~/block-formats/venv
# python3 -m venv --system-site-packages ~/block-formats/venv  # GH200
ln -s ~/block-formats/venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt

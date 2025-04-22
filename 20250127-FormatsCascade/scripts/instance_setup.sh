# Manual steps:
#  scp ~/.gitconfig INSTANCE:~
#  mkdir nethome/USER
#  cd nethome/USER && git clone git@github.com:graphcore-research/squashed-llama.git --branch notebooks SquashedLlama

set -e
set -o xtrace

aws configure

mkdir ~/block-formats || true
python3 -m venv ~/block-formats/venv
ln -s ~/block-formats/venv .venv || true
source .venv/bin/activate
pip install -r requirements.txt

mkdir ~/block-formats/out || true
ln -s ~/block-formats/out out || true

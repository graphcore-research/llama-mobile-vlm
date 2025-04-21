aws configure

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir ~/block-formats-out
ln -s ~/block-formats-out out

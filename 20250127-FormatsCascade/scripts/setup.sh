
echo "Manually:  scp ~/.gitconfig INSTANCE:~"

aws configure

python3 -m venv ~/block-formats-venv
ln -s ~/block-formats-venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir ~/block-formats-out
ln -s ~/block-formats-out out

echo "Start installation"

pip install -r requirements.txt

python -m spacy download en_core_web_sm

echo "Start Downloading pretrained_models"

PRETRAINED_DIR='pretrained_models'

mkdir -p $PRETRAINED_DIR

wget --no-clobber -P $PRETRAINED_DIR https://www.dropbox.com/sh/hrewbpedd2cgdp3/AAA5jwG-P5yxdbFDP4UHvudua/checkpoints/IFRNet/IFRNet_GoPro.pth?dl=0

echo "Done Downloading pretrained_models"

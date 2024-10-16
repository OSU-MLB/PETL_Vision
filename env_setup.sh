conda create -n petl python=3.7
conda activate petl

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

## timm
pip install timm==0.9.12
#
###VTAB
pip install tensorflow==2.11.0
# specifying tfds versions is important to reproduce our results
pip install tfds-nightly==4.4.0.dev202201080107
pip install tensorflow-addons==0.19.0
pip install opencv-python

## CLIP
pip install git+https://github.com/openai/CLIP.git

####utils
pip install dotwiz
pip install pyyaml
pip install tabulate
pip install termcolor
pip install iopath
pip install scikit-learn

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pandas

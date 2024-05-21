mkdir checkpoint

wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz -P dataset
wget https://github.com/DaHui-BT/binaryfacenet/releases/download/v1.0.0/inception-model.pt checkpoint
wget https://github.com/DaHui-BT/binaryfacenet/releases/download/v1.0.0/model.pt checkpoint

tar -zxvf dataset/lfw-funneled.tgz -C ./dataset

pip install -r requirements.txt

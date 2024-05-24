# Binary Face Recognise using Pytorch

this implement on face recognise is base on inception block, the train mode is binary classfication

this model has changed the triple training way to binary classfication way.

## Usage

**step one:** before train / predict this model, you should run the setup.sh firstly
```bash
bash setup.sh
```

**step two:** after doing the step one, then run the ``preprocess.py`` file, this file has extract the face in dataset and save to the path ``dataset/processed``
```bash
python tools/preprocess.py
```

**setp three**: now you can train / predict the model by using the following command
```bash
# train the model
python train.py
# predict the model, the supplyed picture is or not the same person (on facenet)
python binary-predict.py
# predict the model, the supplyed picture is or not the same person (on inception network)
python triple-predict.py
```

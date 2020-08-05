# Alaska 2 Solution

This repository contains solution code for ALASKA 2

# Installation

```bash
pip install -r requirements.txt
```

# Training

```bash
export KAGGLE_2020_ALASKA2=/path/to/alaska2/dataset

# This will take couple of hours to extract DCT matrices from JPEG and save to disk
sh abba_preprocess.sh
sh eugene_preprocess.sh

# This will train models from our ensemble. Requires 4-GPU machine and plenty of time
sh abba_train.sh
sh eugene_train.sh
```

# Inferencing

For sake of convinience, we attach pre-trained models in `models/`, so you may use them right away:

```bash
export KAGGLE_2020_ALASKA2=/path/to/alaska2/dataset

sh abba_predict.sh
sh eugene_predict.sh
```

After running inferencing scripts, final submissions can be found in `submits/` folder. 


# 🏔️ Alaska 2 Solution

This repository contains a prize winning solution code for ALASKA 2. Team ABBA McCandless.

## Key elements

- Trust your CV
- Don't resize, think twice before using hard image augmentations
- Don't use standard image I/O libraries (avoid rounding and clipping pixel values to [0..255])
- Use CNNs without pooling layers in the stem
- Higher resolution for deeper layers is better
- Build a diverse ensemble 
    - EfficientNet 
    - MixNet
    - SRNet 
    - Hand crafted features (DCTR/JRM)

## Documentation

See `./documentation/ABBA_McCandless_documentation.pdf` for the solution documentation. A short description is also available [on the kaggle forum](https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/168546).

## Installation

Eventually run: 
```bash
bash system_requirements.sh
```

And install pip requirements:
```bash
pip install -r requirements.txt
```

## Inferencing

For sake of convinience, we attach pre-trained models in `models/` and `abba/weights/`, so you may use them right away:

```bash
export KAGGLE_2020_ALASKA2=/path/to/alaska2/dataset

sh abba_predict.sh
sh eugene_predict.sh
```

After running inferencing scripts, final submissions can be found in `submits/` folder. 

## Training

```bash
export KAGGLE_2020_ALASKA2=/path/to/alaska2/dataset

# This will take couple of hours to extract DCT matrices from JPEG and save to disk
sh eugene_preprocess.sh

# This will train models from our ensemble. Requires 4-GPU machine and plenty of time
sh abba_train.sh
sh eugene_train.sh
```

## Hardware requirements
Mostly trained on 4xTitan V100 and 3xTitan RTX. 
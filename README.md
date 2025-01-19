# Progressive Denoiser Modification
This project stands for making 'Modified' SURE image for below project.
https://github.com/ArthurFirmino/progressive-denoising

## Dependencies
- python3 (>=3.7)
- numpy
- pytorch
- tqdm
- python3-openimageio
- openexr

## Data
Dataset will be uploaded.

## Code
### Make Dataset
```
python3 SRC/DATASET/MAKE_DATASET.py
```
### Train
```
python3 SRC/MODEL/SD_TRAIN.py
```
### Inference
```
python3 SRC/MODEL/SD_INFER.py
```
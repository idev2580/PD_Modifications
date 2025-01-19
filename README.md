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
Only directory-modified version of progressive denoiser's dataset.
- Train : https://idev2580dev.work/files/TRAIN.tar.gz
- Test : https://idev2580dev.work/files/TEST.tar.gz
- Valid : https://idev2580dev.work/files/VALID.tar.gz

1. Download all these 3 files.
2. Make `DATASET` under this project folder.
3. Extract 3 archives into `DATASET`

After extraction, directory structure should be like this:
```
PD_Modification
|-- DATASET
|   |-- TEST
|   |-- TRAIN
|   |-- VALID
|-- SRC
```


## Code
### Make Dataset
This will generate Per-Pixel Squared Error as ground truth error.
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

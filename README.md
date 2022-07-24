# Official codes for 'Transfer Learning for Color Constancy via Statistic Perspective'

> In this paper, we provide a cool method that can learn scene information from sRGB images via a statistic perspective. 🍺

## Setup Environment

```
% optional
conda create -n anole python=3.7
conda activate anole
% necessary
pip3 install -r requirements.txt
```

## Prepare the dataset

### RAW dataset

#### ==> Option One: Download the pre-processed data. [Only Provide ColorChecker Now]

- Due to the large file size, we only provide [the processed ColorChecker dataset](https://drive.google.com/file/d/1ZXzFWK6iISrajigmeOI2AfE3BGiOGfLn/view?usp=sharing).

#### ==> Option Two: Download the source data and pre-process it locally.

**Step1:** Download the source data and organize files as required.

You need to create the initial folder as follows, and then put the corresponding datasets one by one.

```
data
└── source
    ├── colorchecker2010
    ├── Cube
    └── NUS
```

------

***Dataset1: [ColorChecker (Reporcessed)](https://www2.cs.sfu.ca/~colour/data/shi_gehler/)***

Download the **PNG Images** and **Measured illumination** first. After decompression, the files are organized as follows:

```bash
├── IMG_0901.png
├── IMG_0902.png
├── coordinates
│   ├── IMG_0901_macbeth.txt
│   └── IMG_0902_macbeth.txt
├── img.txt
└── real_illum_568..mat
```

**img.txt**：include all image names.

------

***Dataset2: [NUS-8](https://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html)***

Download **PNG files (ZIP1, ZIP2, ...)**, **MASK**, **GROUNDTRUTH** from eight camera. After decompression, the files are organized as follows:

```
├── Canon1DsMkIII
│   ├── CHECKER
│   │   ├── Canon1DsMkIII_0001_color.txt
│   │   ├── Canon1DsMkIII_0001_mask.txt
│   │   ├── Canon1DsMkIII_0002_color.txt
│   │   └── Canon1DsMkIII_0002_mask.txt
│   ├── Canon1DsMkIII_0001.PNG
│   ├── Canon1DsMkIII_0002.PNG
│   └── Canon1DsMkIII_gt.mat
├── Canon600D
│   ├── CHECKER
│   │   ├── Canon600D_0001_color.txt
│   │   ├── Canon600D_0001_mask.txt
│   │   ├── Canon600D_0002_color.txt
│   │   └── Canon600D_0002_mask.txt
│   ├── Canon600D_0001.PNG
│   ├── Canon600D_0002.PNG
│   ├── Canon600D_CHECKER.zip
│   └── Canon600D_gt.mat
├── FujifilmXM1
...
├── NikonD5200
...
├── OlympusEPL6
...
├── PanasonicGX1
...
├── SamsungNX2000
...
└── SonyA57
```

------

***Dataset3: [Cube/Cube+](https://ipg.fer.hr/ipg/resources/color_constancy)***

Download **PNG files (PNG_1_200.zip, ...)**, **cube+_gt.txt**. After decompression, the files are organized as follows:

```
├── 1.PNG
├── 10.PNG
├── cube+_gt.txt
└── img.txt
```

**img.txt**：include all image names.

------

**Step2:** Pre-process data locally

```bash
python data_preprocess.py --output_dir ./data/processed/ --input_dir ./data/source/ --resize2half False
```

**output_dir**: the path save the processed files (image, illumination, camera_type, mask).

**input_dir**: the path save the source files, as above.

**resize2half**: For speed up training, reduce the length and width of the preprocessed file to half.

### sRGB dataset

A highlight of this paper is that we build an efficient transfer method to leverage the rich scene information from sRGB dataset (Place205, the homepage is [here](http://places.csail.mit.edu/downloadData.html)). We manually selected 14,000+ sRGB images with approximately balanced colors, and the name list is shown in A. Similarly, in order to make training more efficient, we preprocess it to the form of .npy, as RAW datasets do, which you can get it with the following code:

```python
import cv2
import numpy as np2

img_path = ""
img = cv2.imread(img_path)
np.save(output_path, img)
```

## Training

- **Hint**: You need to process the data first!
- Training the model from scratch.

```shell
# Training each data fold step by step.
nohup python main.py --fold_idx 0 > log/TLCC_sota_fold0.log &
nohup python main.py --fold_idx 1 > log/TLCC_sota_fold1.log &
nohup python main.py --fold_idx 2 > log/TLCC_sota_fold2.log &
```

- Then getting the trained model in the directory: `./ckpt/*best.pth`

## Testing 

- After training, it needs to specify the test model for each fold:

```shell
python test.py --data_path --load_ckpt_fold0 --load_ckpt_fold1 --load_ckpt_fold2
```

- Or it also can directly use pretrained model (upload later) to **skip the training step**.

## Citing this work

If you find this code useful for your research, please consider citing the following paper:

```tex
@inproceedings{tang2022transfer,
  title={Transfer Learning for Color Constancy via Statistic Perspective},
  author={Tang, Yuxiang and Kang, Xuejing and Li, Chunxiao and Lin, Zhaowen and Ming, Anlong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```

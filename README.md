# MCoT-MVS

## Installation

Use Python version **3.10.0**:

```bash
conda create -n MCoT-MVS python=3.10.0
conda activate MCoT-MVS
````

Then install the requirements:

```bash
pip install -r requirements.txt
```

---

## Segment

If you want to generate masks and segmentation features manually, follow the steps below:

### 1. Download Checkpoints

* **RAM++**: [ram\_plus\_swin\_large\_14m.pth](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth)
  → Put it in `./segment/ram_models/` directory

* **SAM2**: [sam2.1\_hiera\_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  → Put it in `./segment/sam2/checkpoints/` directory

### 2. Generate Masks

```bash
cd segment
python get_mask.py
```

### 3. Extract Segmentation Features

```bash
python extract_seg_feature_cirr.py
python extract_seg_feature_fiq.py
```

Or download the pre-extracted segmentation features directly:

* **CIRR**: [Download](https://drive.google.com/file/d/1MV8ITivi-Ik-qehj5Ud3aw11pG-cIp9V/view?usp=drive_link)
* **FashionIQ**: [Download](https://drive.google.com/file/d/19eBDQEFDpzSEALGSnm6lgSY31MJiYsLz/view?usp=drive_link)


---

## Data Structure

### FashionIQ

```
data/
└── fiq/
    ├── fashionIQ_dataset/
    │   ├── correction_dict_*.json
    │   ├── keywords_in_mods_*.json
    │   ├── captions/
    │   ├── images/
    │   ├── resized_image/
    │   └── image_splits/
    ├── llm_data/
    └── segment/seg_features_vit-h_patch/
```

> 📥 Download FashionIQ images from [Yandex Disk](https://disk.yandex.com/d/Z2E54WCwvrQA3A)
> Then run the [resize\_images.py](https://github.com/XiaoxiaoGuo/fashion-iq/blob/master/start_kit/resize_images.py) script.
> Resized images should be saved to `./data/fiq/fashionIQ_dataset/resized_image/`.

---

### CIRR

```
data/
└── cirr_dataset/
    ├── train/
    ├── dev/
    ├── test1/
    ├── cirr/
    │   ├── captions/
    │   └── image_splits/
    ├── llm_data/
    └── segment/seg_features_vit-h_patch/
```

> 📥 Follow the [official CIRR repository](https://github.com/Cuberick-Orion/CIRR) to download and unzip the dataset.
> Place the files under `./data/cirr_dataset/`.

---

## Training

The training code will be made public after the paper is accepted. 
---

## Inference

Download pre-trained checkpoints:

* **CIRR checkpoint**: [Download](https://drive.google.com/file/d/1OJRFIk1TTwd4iRgNNcpBESrNMJuUVo2h/view?usp=drive_link)
* **dress checkpoint**: [Download](https://drive.google.com/file/d/1tNMtH1SABWU7HsREc297STgkj52gtsFL/view?usp=drive_link)
* **shirt checkpoint**: [Download](https://drive.google.com/file/d/1r526gq_lHyDxm1fhrTxDwsFzChDNNTf3/view?usp=drive_link)
* **toptee checkpoint**: [Download](https://1drv.ms/u/c/a76cc9a14c451e3b/EatrK0L9tc9IqboXxSN76DcBwIoXtuDosRsMkn7XRWiFeg?e=t2ZxuS)
  → Place in the `checkpoints/` directory

### CIRR Inference

```bash
cd src
CUDA_VISIBLE_DEVICES=<device_id> python cirr_test_submission.py --model_path checkpoints/your_model.pt --submission_name test
```

### FashionIQ Inference

```bash
cd src
CUDA_VISIBLE_DEVICES=<device_id> python fiq_validate.py --model_path checkpoints/your_model.pt --dress_type dress
```
# VRD-IR: Visual Recognition-Driven Image Restoration for Multiple Degradation (CVPR2023)

## Introduction
This repository contains the implementation of the **VRD-IR (Visual Recognition-Driven Image Restoration)** model proposed in the CVPR 2023 paper *"Visual Recognition-Driven Image Restoration for Multiple Degradation with Intrinsic Semantics Recovery"*. VRD-IR focuses on restoring recognition-friendly high-quality images from multiple unknown degradations (e.g., haze, noise, rain) and can be directly integrated into downstream vision tasks (classification, detection, person Re-ID) as an enhancement module.

## Environment Setup
    conda env create -f environment.yaml
    conda activate vrdir

## Dataset
Please download the RESIDE, BSD400, WED, and Rain100L datasets, and organize them as follows respectively. For more details, please refer to the experimental section and appendices in the paper.

```
dataset/
├── train/
│   ├── degraded/    
│   └── clean/     
├── val/
│   ├── degraded/
├── └── clean/
```
Note: During model training or testing, you can execute the `generate-path.py` file in the `data` directory, pass in the paths of degraded images and clean images to generate a paired txt file, then pass the path of this txt file to the `--deg_dir` and `--clean_dir` parameters in the `train.py`, `pretrain_DNC.py`, `pretrain_SAD.py` file. The same applies to the validation set.

Generate a paired txt file：
```bash
python generate_path.py \
    --gt_dir /path/to/clean_images \
    --deg_dir /path/to/degraded_images \
    --output SOTS-pairs.txt
```

The format of the generated paired txt file is as follows:
```
degraded_image_path|clean_image_path
```

## Training

Training follows a three-stage pipeline. All checkpoints and visualization results are saved to `--work_dir`.

### Stage 1: Pre-train SAD
Use `pretrain_SAD.py` to train SAD. 

```bash
python pretrain_SAD.py \
    --clean_dir /path/to/train/clean \
    --deg_dir /path/to/train/deg \
    --val_clean_dir /path/to/val/clean \
    --val_deg_dir /path/to/val/deg \
    --work_dir ./SAD_pre/results/version_1 \
```

### Stage 2: Pre-train Multiple DNC Branches
Use `pretrain_DNC.py` to train 1 DNC branch per degradation type respectively (e.g. 1 for haze, 1 for noise, 1 for rain).

```bash
python pretrain_DNC.py \
    --clean_dir /path/to/train/clean \
    --deg_dir /path/to/train/deg \
    --val_clean_dir /path/to/val/clean \
    --val_deg_dir /path/to/val/deg \
    --sad_ckpt ./SAD_pre/results/version_1/checkpoint/checkpoint_epoch30.pth \
    --work_dir ./DNC_pre/results/hazy/version_1 \
    --branch_name hazy \
```
Repeat for other degradations. Train 2 more DNC branches.

### Stage 3: Train ISE
Use `train.py` to train ISE, which fuses multi-branch DNC outputs via FGM. The pre-trained SAD and DNC branches are fixed as prior knowledge.

```bash
python pretrain_DNC.py \
    --clean_dir /path/to/train/clean \
    --deg_dir /path/to/train/deg \
    --val_clean_dir /path/to/val/clean \
    --val_deg_dir /path/to/val/deg \
    --pre_sad_checkpoint ./SAD_pre/results/version_1/checkpoint/checkpoint_epoch30.pth \
    --pre_dnc1_checkpoint ./DNC_pre/results/hazy/version_1/checkpoint/dnc_branch_hazy_epoch30.pth \
    --pre_dnc2_checkpoint ./DNC_pre/results/noisy/version_1/checkpoint/dnc_branch_noisy_epoch30.pth \
    --pre_dnc3_checkpoint ./DNC_pre/results/rainy/version_1/checkpoint/dnc_branch_rainy_epoch30.pth \ 
    --work_dir ./results/version_1 \
```
## Testing
The test process also uses the `train.py` file, requiring only minor modifications.
1. Prepare the test dataset and the pre-trained SAD and ISE models.
2. Add the `--resume_from` parameter to pass in the path of the pre-trained ISE model. Pass the address of the test dataset to the `--val_deg_dir` and `--val_clean_dir` parameters.
3. Comment out line 88 `trainer.train(train_sampler)` in the `train.py` file, and use `# trainer.test()` on line 90 for testing.
4. The test command line is the same as that of the training phase Stage 3.




If you have any time, please email hj0117@mail.ustc.edu.cn 

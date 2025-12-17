# COMP576 – Lane Segmentation with YOLOv11

This repository contains experiment scripts and configuration files for the COMP 576 final project on lane detection using YOLOv11-based segmentation models. The project focuses on analyzing the impact of training configurations and data augmentation strategies on segmentation stability and performance using the CULane dataset.

## Experiment Workflow
The baseline model and the initial experiments in this project were conducted via command-line execution. Subsequent parameter search and evaluation were performed using scripted workflows. We began with a baseline YOLOv11 segmentation configuration and gradually explored different training parameter settings to improve model stability and segmentation quality.

### Baseline and Initial Experiments
- A default YOLOv11 segmentation model was first trained as the baseline.
- An initial modified configuration was then tested by adjusting mask generation and data augmentation parameters.
- These early experiments helped identify key parameters affecting loss variance and segmentation stability.

### Parameter Search and Trial Selection
- Based on observations from the initial experiments, we designed a systematic parameter search involving 10 different training configurations.
- Each configuration was evaluated using short training runs and multiple metrics, including validation loss, loss variance, and detection performance.
- Among these trials, the configuration labeled as **Trial-7** achieved the best overall performance under the custom evaluation metric.

### Example Command-Line Usage

The baseline model and the initial parameter trials were executed via command-line training. Below is an example command used during these early experiments:

yolo segment train model=yolo11s-seg.pt data=lane-seg.yaml imgsz=640 epochs=20 batch=2 mask_ratio=2 hsv_h=0.015 hsv_s=0.5 hsv_v=0.4 perspective=0.0008 scale=0.2 translate=0.2 mixup=0.0 copy_paste=0.05 flipud=0 fliplr=0.5 erasing=0.15

### Final Training
- The selected Trial-7 configuration was then trained for a longer schedule using the main training script (`576_training.py`).
- The final comparison was conducted between the baseline model, the previously identified best configuration, and the Trial-7 configuration.

## Repository Contents

- `576_training.py`: Main training script used for baseline, intermediate, and final experiments.
- `lane-seg.yaml`: YAML configuration file specifying training parameters and data augmentation settings.
- `make_culane_subset.py`: Script for constructing a reduced CULane training and validation subset.
- `plot_results.py`: Script for visualizing validation loss curves and stability metrics used in the final report.

## Notes
- The CULane dataset is not included due to size and licensing constraints.
- Model weights and raw training logs are not uploaded.
- Key quantitative and qualitative results are reported in the final project paper.

## Course Information
COMP 576 – Deep Learning  
Rice University
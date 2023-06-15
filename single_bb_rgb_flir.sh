#!/bin/bash

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
--rgb-checkpoint-path /groups/ARCL/flir_aligned_backbones/flir-rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee single_bb_rgb_flir.txt

echo "" | tee -a single_bb_rgb_flir.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
--rgb-checkpoint-path /groups/ARCL/flir_aligned_backbones/flir-rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_flir.txt

echo "" | tee -a single_bb_rgb_flir.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
--rgb-checkpoint-path /groups/ARCL/flir_aligned_backbones/flir-rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_flir.txt
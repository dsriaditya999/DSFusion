#!/bin/bash

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_overcast \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_challenge \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

##############################################################################################################
echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_overcast \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_challenge \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

##############################################################################################################

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_overcast \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_challenge \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


##############################################################################################################

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_overcast \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_challenge \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


##############################################################################################################

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_overcast \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt


echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_challenge \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt

echo "" | tee -a fixed_fusion_cbam.txt

python validate_all_att.py /groups/ARCL/sdeevi/M3FD_Detection --dataset m3fd_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
--classwise --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
--thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam.txt
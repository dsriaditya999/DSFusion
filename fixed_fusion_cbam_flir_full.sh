#!/bin/bash

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

#####################################################################################################################

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

##############################################################################################################

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
--classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt


##############################################################################################################



# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_DAY_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# #####################################################################################################################

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_NIGHT_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# ##############################################################################################################

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_day \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_night \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt

# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt

# python validate_all_att.py /home/sdeevi/Research/Datasets/FLIR_Aligned --dataset flir_aligned_full \
# --checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir/EXP_RGBHEAD_FLIR_ALIGNED_FULL_CBAM/last.pth.tar \
# --classwise --split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
# --thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_before_fusion_cbam_flir_full.txt


# echo "" | tee -a fixed_before_fusion_cbam_flir_full.txt


# ##############################################################################################################


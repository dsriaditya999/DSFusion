#!/bin/bash

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_DAY_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

#####################################################################################################################
echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt

echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_stf/EXP_RGBHEAD_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_stf_pascal.txt
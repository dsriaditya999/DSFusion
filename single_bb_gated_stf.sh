#!/bin/bash

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt

echo "" | tee -a single_bb_gated_stf.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--thermal-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/gated/model_best.pth.tar --init-fusion-head-weights thermal \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch thermal | tee -a single_bb_gated_stf.txt
#!/bin/bash

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_full \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_day \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_clear_night \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_day \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_fog_night \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_day \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt

echo "" | tee -a single_bb_rgb_stf_pascal.txt

python validate_all_att.py /groups/ARCL/rgb_gated_aligned --dataset stf_snow_night \
--rgb-checkpoint-path /groups/ARCL/stf_1280_all_scenes_weights/rgb/model_best.pth.tar --init-fusion-head-weights rgb \
--classwise --split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=8 --branch rgb | tee -a single_bb_rgb_stf_pascal.txt
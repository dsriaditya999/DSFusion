#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

# python validate_all_att_cls.py /media/hdd2/rgb_gated_aligned --dataset stf_full --num-scenes 7 \
# --checkpoint /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# --checkpoint-cls /home/ganlu/workspace/efficientdet-pytorch/output/train/stf-rgb-backbone-cls/model_best.pth.tar \
# --checkpoint-scenes /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
# --split test --num-classes 4 \
# --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_cls.txt

# echo "" | tee -a fixed_fusion_cbam_stf_cls.txt


# python validate_all_att.py /media/hdd2/rgb_gated_aligned --dataset stf_full \
# --checkpoint /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# --split test --num-classes 4 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_pascal.txt

# echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


# export CUDA_VISIBLE_DEVICES=1

######################################### OLD #################################################
# python validate_all_att_cls.py /media/hdd2/rgb_gated_aligned --dataset stf_snow_rain --num-scenes 7 \
# --checkpoint /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# --checkpoint-cls /home/ganlu/workspace/efficientdet-pytorch/output/train/stf-rgb-backbone-cls-old/model_best.pth.tar \
# --checkpoint-scenes /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FOG_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FOG_NIGHT_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_SNOW_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_SNOW_NIGHT_CBAM/model_best.pth.tar \
# --split test --num-classes 4 \
# --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_cls.txt

# echo "" | tee -a fixed_fusion_cbam_stf_cls.txt


# python validate_all_att.py /media/hdd2/rgb_gated_aligned --dataset stf_snow_rain \
# --checkpoint /home/ganlu/trained_models_dsf/stf/output/train_stf/EXP_STF_FULL_CBAM/model_best.pth.tar \
# --split test --num-classes 4 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_pascal.txt

# echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt


####################################### NEW #################################################
python validate_all_att_cls.py /media/hdd2/rgb_gated_aligned --dataset stf_clear --num-scenes 3 \
--checkpoint /home/ganlu/trained_stf_clear/EXP_CLEAR3_STF_CLEAR_CBAM/model_best.pth.tar \
--checkpoint-cls /home/ganlu/workspace/efficientdet-pytorch/output/train/stf-rgb-backbone-cls/model_best.pth.tar \
--checkpoint-scenes /home/ganlu/trained_stf_clear/EXP_CLEAR3_STF_CLEAR_CBAM/model_best.pth.tar \
/home/ganlu/trained_stf_clear/EXP_CLEAR3_STF_CLEAR_DAY_CBAM/model_best.pth.tar \
/home/ganlu/trained_stf_clear/EXP_CLEAR_STF_CLEAR_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 4 \
--model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_cls.txt

echo "" | tee -a fixed_fusion_cbam_stf_cls.txt


# python validate_all_att.py /media/hdd2/rgb_gated_aligned --dataset stf_clear \
# --checkpoint /home/ganlu/trained_stf_clear/EXP_CLEAR3_STF_CLEAR_CBAM/model_best.pth.tar \
# --split test --num-classes 4 \
# --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam_stf_pascal.txt

# echo "" | tee -a fixed_fusion_cbam_stf_pascal.txt

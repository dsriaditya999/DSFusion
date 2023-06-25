# #!/bin/bash
export CUDA_VISIBLE_DEVICES=1

################################################# M3FD #####################################################
# python validate_all_att_cls.py /home/carson/data/m3fd --dataset m3fd_full --num-scenes 5 \
# --checkpoint /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
# --checkpoint-cls /home/ganlu/workspace/efficientdet-pytorch/output/train/m3fd-rgb-backbone-cls/model_best.pth.tar \
# --checkpoint-scenes /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_DAY_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_NIGHT_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_OVERCAST_CBAM/model_best.pth.tar \
# /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_CHALLENGE_CBAM/model_best.pth.tar \
# --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
# --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
# --classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam | tee fixed_fusion_cbam_m3fd_cls.txt

# echo "" | tee -a fixed_fusion_cbam_m3fd_cls.txt


# python validate_all_att.py /home/carson/data/m3fd --dataset m3fd_full \
# --checkpoint /home/ganlu/trained_models_dsf/m3fd/output/train_m3fd/EXP_M3FD_FULL_CBAM/model_best.pth.tar \
# --split test --num-classes 6 --rgb_mean 0.49151019 0.50717567 0.50293698 --rgb_std 0.1623529 0.14178433 0.13799928 \
# --thermal_mean 0.33000296 0.33000296 0.33000296 --thermal_std 0.18958051 0.18958051 0.18958051 \
# --classwise --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee fixed_fusion_cbam_m3fd.txt

# echo "" | tee -a fixed_fusion_cbam_m3fd.txt


############################################### FLIR #####################################################
python validate_all_att_cls.py /home/ganlu/workspace/FLIR_Aligned --dataset flir_aligned_full --num-scenes 3 \
--checkpoint  /home/ganlu/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar \
--checkpoint-cls /home/ganlu/workspace/efficientdet-pytorch/output/train/flir-rgb-backbone-cls/classifier.pth.tar \
--checkpoint-scenes /home/ganlu/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar \
/home/ganlu/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_DAY_CBAM/model_best.pth.tar \
/home/ganlu/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_NIGHT_CBAM/model_best.pth.tar \
--split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--classwise --model efficientdetv2_dt --batch-size=1 --branch fusion --att_type cbam | tee fixed_fusion_cbam_flir_cls.txt

echo "" | tee -a fixed_fusion_cbam_flir_cls.txt


python validate_all_att.py /home/ganlu/workspace/FLIR_Aligned --dataset flir_aligned_full \
--checkpoint  /home/ganlu/train_flir_before/EXP_BeforeBiFPN_FLIR_ALIGNED_FULL_CBAM/model_best.pth.tar \
--split test --num-classes 3 --rgb_mean 0.62721553 0.63597459 0.62891984 --rgb_std 0.16723704 0.17459581 0.18347738 \
--thermal_mean 0.53584253 0.53584253 0.53584253 --thermal_std 0.24790472 0.24790472 0.24790472 \
--classwise --model efficientdetv2_dt --batch-size=8 --branch fusion --att_type cbam | tee -a fixed_fusion_cbam_flir.txt


echo "" | tee -a fixed_fusion_cbam_flir.txt
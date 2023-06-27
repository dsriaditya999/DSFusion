python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_day \
--checkpoint "/home/sdeevi/Research/deep-sensor-fusion-main/Pretrained Model Files/model_best.pth.tar" \
--num-classes 90 --model efficientdetv2_dt --branch thermal --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee fusion_cbam_dbb_flir_thermal.txt

echo "" | tee -a fusion_cbam_dbb_flir_thermal.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_night \
--checkpoint "/home/sdeevi/Research/deep-sensor-fusion-main/Pretrained Model Files/model_best.pth.tar" \
--num-classes 90 --model efficientdetv2_dt --branch thermal --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a fusion_cbam_dbb_flir_thermal.txt

echo "" | tee -a fusion_cbam_dbb_flir_thermal.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_full \
--checkpoint "/home/sdeevi/Research/deep-sensor-fusion-main/Pretrained Model Files/model_best.pth.tar" \
--num-classes 90 --model efficientdetv2_dt --branch thermal --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a fusion_cbam_dbb_flir_thermal.txt

echo "" | tee -a fusion_cbam_dbb_flir_thermal.txt
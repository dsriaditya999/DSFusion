python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132722-EXP_FULL_AFTER_CBAM_DAY/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132722-EXP_FULL_AFTER_CBAM_DAY/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132722-EXP_FULL_AFTER_CBAM_DAY/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132630-EXP_FULL_AFTER_CBAM_NIGHT/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132630-EXP_FULL_AFTER_CBAM_NIGHT/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132630-EXP_FULL_AFTER_CBAM_NIGHT/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

#
python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_day \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132743-EXP_FULL_AFTER_CBAM_FULL/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_night \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132743-EXP_FULL_AFTER_CBAM_FULL/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

python validate_cbam.py /home/sdeevi/Research/Datasets/FLIR_Aligned/ --dataset flir_aligned_full \
--checkpoint /home/sdeevi/Research/deep-sensor-fusion-main/output/train_flir_after_diffbb/20230626-132743-EXP_FULL_AFTER_CBAM_FULL/model_best.pth.tar \
--num-classes 90 --model efficientdetv2_dt --branch fusion --classwise --split test \
--mean 0.519 0.519 0.519 --std 0.225 0.225 0.225 | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt

echo "" | tee -a afterbifpn_fusion_cbam_dbb_flir_full_best.txt
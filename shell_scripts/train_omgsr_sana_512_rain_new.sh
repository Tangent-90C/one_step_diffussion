export HF_ENDPOINT=https://hf-mirror.com
# 48G 显存可以跑通1024x1024的omgsr+sana模型训练
# python train/train_omgsr_sana.py --config ./configs/omgsr_sana_1024.yml # 原始跑法，这个大概要47G显存
accelerate launch --config_file ./configs/单机ZeRO2.yaml train/train_omgsr_sana_rain_lightning.py --config ./configs/omgsr_sana_512_rain.yml # 启用了ZeRO2优化，这个大概要44G显存



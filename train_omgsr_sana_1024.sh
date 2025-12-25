export HF_ENDPOINT=https://hf-mirror.com
# accelerate launch --config_file ./configs/accelerate_config.yaml train/train_omgsr_sana.py --config ./configs/omgsr_sana_1024.yml
# python train/train_omgsr_sana.py --config ./configs/omgsr_sana_1024.yml
accelerate launch --config_file ./configs/default_config.yaml train/train_omgsr_sana.py --config ./configs/omgsr_sana_1024.yml



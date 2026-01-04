# python infer/infer_omgsr_sana_rain.py \
#     --input_image /mnt/HDD-data/jianuo/dataset/GT-Rain/GT-RAIN_val \
#     --output_dir ./experiments_omgsr_sana_rain \
#     --model_path /mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers \
#     --lora_path /home/chenjn/OMGSR/omgsr_trainings/omgsr_sana_1024_rain/weight-12000 \
#     --process_size 1024 \
#     --mid_timestep 244 \
#     --guidance 5.0 \
#     --weight_dtype bf16 \
#     --prompt "" \
#     --device cuda:0


python infer/infer_omgsr_sana_rain.py \
    --input_image /mnt/HDD-data/jianuo/dataset/GT-Rain/GT-RAIN_test \
    --output_dir ./experiments_omgsr_sana_rain/test \
    --model_path /mnt/HDD-data/jianuo/models/Sana_Sprint_1.6B_1024px_diffusers \
    --lora_path /home/chenjn/OMGSR/omgsr_trainings/omgsr_sana_1024_rain/weight-12000 \
    --process_size 1024 \
    --mid_timestep 244 \
    --guidance 5.0 \
    --weight_dtype bf16 \
    --prompt "" \
    --device cuda:0

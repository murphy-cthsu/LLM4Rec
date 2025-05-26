# python src/training_new.py \
#     --dataset "user_session_data" \
#     --lambda_V 0.1 \
#     --data_path "./data" \
#     --pretrained_path "./pretrained" \
#     --model_name "Qwen/Qwen3-1.7B" 

python src/finetuning_new.py \
    --dataset "user_session_data" \
    --lambda_V 0.1 \
    --model_path "models"\
    --model_name "Qwen/Qwen3-1.7B" 
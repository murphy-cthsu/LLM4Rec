# for i in {0..9}
# do
#     python src/training_new.py \
#         --dataset "user_session_data_${i}" \
#         --lambda_V 0.1 \
#         --data_path "./data" \
#         --pretrained_path "./pretrained" \
#         --model_name "Qwen/Qwen3-1.7B"

#     python src/finetuning_new.py \
#         --dataset "user_session_data_${i}" \
#         --lambda_V 0.1 \
#         --model_path "models" \
#         --model_name "Qwen/Qwen3-1.7B"
# done
python3 src/training_new.py \
    --num_data 1 \
    --lambda_V 0.1 \
    --data_path "./data" \
    --model_name "Qwen/Qwen3-1.7B"
python3 src/finetuning_new.py \
    --num_dataset 1 \
    --lambda_V 0.1 

python3 src/predict.py \
    --dataset "user_session_data_0" \
    --lambda_V 0.1 \
    --data_path "data"  \
    

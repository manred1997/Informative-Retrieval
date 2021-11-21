export lr=1e-5
export s=3407
echo "${lr}"
export MODEL_DIR=phobert
echo "${MODEL_DIR}"
python3 main.py --token_level word-level --model_type phobert --model_dir $MODEL_DIR --data_dir ../data --seed $s --do_train --do_eval --save_steps 300 --logging_steps 300 --train_batch_size 8 --eval_batch_size 32 --gpu_id 0 --learning_rate $lr --num_train_epochs 20 --early_stopping 20
CUDA_VISIBLE_DEVICES=0, python run_clm_kv_compression.py \
    --model_name_or_path $1 \
    --train_file wikitext2/train.json \
    --validation_file wikitext2/test.json \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --per_device_eval_batch_size 4 \
    --block_size 256 \
    --preprocessing_num_workers 12 \
    --output_dir ./output/${1}/kv_${2}_${3} \
    --compress \
    --max_span_length $3 \
    --bound_ratio $2 \
    --r 16 \
    --seed 0 \
    --learning_rate 2e-5

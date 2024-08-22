export MODEL_PATH='/dev/shm/Mistral-7B-v0.1'
# export MODEL_NAME=Meta-Llama-3-8B

export DATA_PATH=$1
export SAVE_PATH=$2
export LOGGING_DIR=$3
export NUM_TRAIN_EPOCHS=$4
export QUANT_BIT=$5
export KBIT=$6
export VBIT=$7
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --include localhost:4,5,6,7\
     train4attn.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --model_max_length 32768 \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 2500 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 15 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits ${QUANT_BIT} \
    --kbit ${KBIT} \
    --vbit ${VBIT} \
    --quant_type int2-asym \
    --q_group_size 32 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip None

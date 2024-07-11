
export MODEL_PATH=NousResearch/Meta-Llama-3-8B
export MODEL_NAME=Meta-Llama-3-8B

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

export NUM_GPUS=4

# --clip BitDistiller/quantization/clip_cache/WizardCoder-7B/7b-int2-g128-twoclip.pt
# --evaluation_strategy "steps"
# --eval_steps 4
deepspeed --num_gpus=${NUM_GPUS} train4attn.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --model_max_length 8192 \
    --output_dir ${SAVE_PATH} \
    --logging_dir ${LOGGING_DIR} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 400 \
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

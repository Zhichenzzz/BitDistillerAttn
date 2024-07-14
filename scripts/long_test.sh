# model e.g.: meta-llama/Llama-2-7b-hf
cd test/longbench

gpuid=$1
model=$2
k_bits=$2
v_bits=$3
group_size=$4


CUDA_VISIBLE_DEVICES=$gpuid python pred_longbench.py --model $model \
    --maxlen 8192 \
    --quantize_k True \
    --quantize_v True \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size


cd train/


bash train.sh ../data/generation/datasets/Mistral-7B-v0.1/mix_wiki_alpaca_T0.7_N1024_S42_8000.json  /dev/shm/zhichenz/ckpts/Mistral-7B-v0.1/k2v1g32/ ./logs/Mistral-7B-v0.1/k2v1g32  4 2 2 1

bash train.sh ../data/generation/datasets/Mistral-7B-v0.1/mix_wiki_alpaca_T0.7_N1024_S42_8000.json  /dev/shm/zhichenz/ckpts/Mistral-7B-v0.1/k2v2g32/ ./logs/Mistral-7B-v0.1/k2v2g32  4 2 2 2


cd ~/zhichenz/BitDistillerAttn/test/longbench

CUDA_VISIBLE_DEVICES=6,7 python pred_longbench.py --model Mistral-7B-ft21 --maxlen 32768 --quantize_k True --quantize_v True --kbit 2 --vbit 1 --sparsity_ratio 0.0 --group_size 32

CUDA_VISIBLE_DEVICES=4,5,6,7 python pred_longbench.py --model Mistral-7B-ft22 --maxlen 32768 --quantize_k True --quantize_v True --kbit 2 --vbit 2 --sparsity_ratio 0.0 --group_size 32
cd train/

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B/mix_wiki_alpaca_8000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B/k1v1-g16/ /home/superbench/zhichenz/BitDistillerAttn/train/logs/Meta-Llama-3-8B/k1v1-g16/ 4 1 1 1

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B/mix_wiki_alpaca_8000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B/k2v2-g16/ /home/superbench/zhichenz/BitDistillerAttn/train/logs/Meta-Llama-3-8B/k2v2-g16/ 4 2 2 2

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B/mix_wiki_alpaca_8000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B-Instruct/k2v2-g32 /home/superbench/zhichenz/BitDistillerAttn/train/logs/Meta-Llama-3-8B-Instruct/k2v2-g32 4 2 2 2


python generate_vllm.py --base_model /dev/shm/Meta-Llama-3-8B/ --dataset_name wikitext --out_path ./datasets/Meta-Llama-3-8B/ --max_sample 3000

python generate_vllm.py --base_model /dev/shm/Mistral-7B-v0.1 --dataset_name wikitext --out_path ./datasets/Mistral-7B-v0.1 --max_sample 3000

python generate_vllm.py --base_model /dev/shm/Meta-Llama-3-8B/ --dataset_name alpaca --out_path ./datasets/Meta-Llama-3-8B/ --max_sample 5000

python generate_vllm.py --base_model /dev/shm/Mistral-7B-v0.1 --dataset_name longalpaca --out_path ./datasets/Mistral-7B-v0.1 --max_sample 2000

python generate_vllm.py --base_model /dev/shm/Meta-Llama-3-8B-Instruct/ --dataset_name longalpaca --out_path ./datasets/Meta-Llama-3-8B-Instruct/ --max_sample 3000

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B/mix_long_alpaca_6000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B/test ./logs/Meta-Llama-3-8B/int2-g32-test/ 4

bash scripts/long_test.sh 2,3,4,5,6,7 Meta-Llama-3-8B-Instruct 2 2 32

CUDA_VISIBLE_DEVICES=0,1,2,3

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B-Instruct/mix_long_alpaca_6000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B-Instruct/k2v2g32-long/ ./logs/Meta-Llama-3-8B-Instruct/k2v2g32-long 4 2 2 2

bash train.sh ../data/generation/datasets/Mistral-7B-v0.1/mix_wiki_alpaca_T0.7_N1024_S42_8000.json  /dev/shm/zhichenz/ckpts/Mistral-7B-v0.1/k2v1g32/ ./logs/Mistral-7B-v0.1/k2v1g32 4 2 2 1
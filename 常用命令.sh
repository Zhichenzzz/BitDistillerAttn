python generate_vllm.py --base_model /dev/shm/Meta-Llama-3-8B/ --dataset_name wikitext --out_path ./datasets/Meta-Llama-3-8B/ --max_sample 3000

python generate_vllm.py --base_model /dev/shm/Meta-Llama-3-8B/ --dataset_name alpaca --out_path ./datasets/Meta-Llama-3-8B/ --max_sample 5000

bash train.sh ../data/generation/datasets/Meta-Llama-3-8B/mix_wiki_alpaca_8000.json /dev/shm/zhichenz/ckpts/Meta-Llama-3-8B/test ./logs/Meta-Llama-3-8B/int2-g32-test/ 4

bash scripts/long_test.sh 2,3,4,5,6,7 Meta-Llama-3-8B-Instruct 2 2 32
import json
import random

all_outputs = []

json_path1 = "./datasets/Meta-Llama-3-8B/wikitext_T0.7_N1024_S42_3000.json"
json_path2 = "./datasets/Meta-Llama-3-8B/alpaca_T0.7_N1024_S42_5000.json"

with open(json_path1, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

with open(json_path2, 'r') as f:
    dataset_for_eval = f.readlines()
for line in dataset_for_eval:
    json_data = json.loads(line)
    all_outputs.append(json_data)

random.shuffle(all_outputs)

with open('./datasets/Meta-Llama-3-8B/mix_wiki_alpaca_8000.json', 'w') as f:
    for item in all_outputs:
        f.write(json.dumps(item) + '\n')

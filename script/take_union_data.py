import os
import json
import random

file_path = "D:\Gradient_based_ES_for_Attack_MedVLM\evaluate_result"

union_indexs = [set(), set()]
for file_name in os.listdir(file_path):
    json_path = os.path.join(file_path, file_name)
    with open(json_path, 'r') as f:
        data = json.load(f)
        print(len(data))
        for sample in data:
            union_indexs[sample['gt_id']].add(sample['id'])

sample_0 = random.sample(list(union_indexs[0]), 500)
sample_1 = random.sample(list(union_indexs[1]), 500)    
with open("union_data.txt", 'w') as f:
    for line in sample_0 + sample_1:
        f.write(str(line) + '\n')
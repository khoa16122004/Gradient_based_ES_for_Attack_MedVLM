import os
import json
import random

file_path = r"D:\Gradient_based_ES_for_Attack_MedVLM\evaluate_result"

indexs = [None, None]  # để biết file đầu tiên

for file_name in os.listdir(file_path):
    json_path = os.path.join(file_path, file_name)
    with open(json_path, 'r') as f:
        data = json.load(f)

    current_samples = [set(), set()]
    for sample in data:
        current_samples[sample['gt_id']].add(sample['id'])

    if indexs[0] is None:
        indexs[0] = current_samples[0]
        indexs[1] = current_samples[1]
    else:
        indexs[0] &= current_samples[0]
        indexs[1] &= current_samples[1]

# convert sang list
index_0 = list(indexs[0])
index_1 = list(indexs[1])

sample_0 = random.sample(index_0, min(500, len(index_0)))
sample_1 = random.sample(index_1, min(500, len(index_1)))

final = sample_0 + sample_1

with open("intersection.txt", "w") as f:
    for x in final:
        f.write(str(x) + "\n")

import json

with open('D:\Gradient_based_ES_for_Attack_MedVLM\evaluate_result\ViT-B-32.json', 'r') as f:
    results = json.load(f)

acc = [0, 0]
total = [0, 0]
for sample in results:
    if sample['gt_id'] == sample['pred_id']:
        acc[sample['gt_id']] += 1
    total[sample['gt_id']] += 1


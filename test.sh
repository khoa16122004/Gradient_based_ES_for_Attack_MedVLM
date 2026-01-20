python test_clean_performance.py \
    --dataset_name rsna \
    --model_name medclip \
    --batch_size 64

python test_clean_performance.py \
    --dataset_name rsna \
    --model_name biomedclip \
    --batch_size 64

python test_clean_performance.py \
    --dataset_name rsna \
    --model_name ViT-B-32 \
    --batch_size 64
    
python test_clean_performance.py \
    --dataset_name rsna \
    --model_name ViT-B-16 \
    --batch_size 64
python test_clean_performance.py \
    --dataset_name rsna \
    --model_name ViT-L-14 \
    --batch_size 64
python train.py \
    -s data/lerf_mask/figurines -r 1 \
    -m output/verify_figurines_uncertain \
    --config_file config/gaussian_dataset/train_aniso_uncertain.json \
    --train_split

# 渲染 + 评测
python render_lerf_mask.py -m output/verify_figurines_uncertain \
    --skip_train --num_classes 256 --images images

python tests/eval_compare.py \
    --scene figurines \
    --baseline_model output/verify_figurines_baseline \
    --ours_model output/verify_figurines_uncertain \
    --iteration 30000 \
    --out_json output/verify_logs/metrics_figurines_uncertain.json
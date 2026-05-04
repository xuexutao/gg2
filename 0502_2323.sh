# python train.py \
#     -s data/lerf_mask/figurines -r 1 \
#     -m output/verify_figurines_uncertain \
#     --config_file config/gaussian_dataset/train_aniso_uncertain.json \
#     --train_split

# # 渲染 + 评测
# python render_lerf_mask.py -m output/verify_figurines_uncertain \
#     --skip_train --num_classes 256 --images images

# python tests/eval_compare.py \
#     --scene figurines \
#     --baseline_model output/verify_figurines_baseline \
#     --ours_model output/verify_figurines_uncertain \
#     --iteration 30000 \
#     --out_json output/verify_logs/metrics_figurines_uncertain.json


# # Baseline
# SCENE=teatime
# # Baseline
# # python train.py -m output/${SCENE}_baseline --config_file config/gaussian_dataset/train.json --source_path data/lerf_mask/${SCENE}
# python render_lerf_mask.py -m output/${SCENE}_baseline --skip_train --num_classes 256 --images images
# # Aniso-only
# # python train.py -m output/${SCENE}_aniso_only --config_file config/gaussian_dataset/train_aniso_only.json --source_path data/lerf_mask/${SCENE}
# python render_lerf_mask.py -m output/${SCENE}_aniso_only --skip_train --num_classes 256 --images images
# # Full (aniso + normal)
# # python train.py -m output/${SCENE}_full --config_file config/gaussian_dataset/train_aniso.json --source_path data/lerf_mask/${SCENE}
# python render_lerf_mask.py -m output/${SCENE}_full --skip_train --num_classes 256 --images images


python script/data_prepare.py -s data/room \
    --sam_checkpoint /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth \
    --groundingdino_config /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py \
    --groundingdino_checkpoint /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/51/new_workspace/gg2/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth \
    --text_prompts "sofa. TV. keyboard. mouse. drinking glass. armchair"
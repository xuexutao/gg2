rm -rf data/room/images_train
rm -rf data/room/test_mask
rm data/room/images/test_*.jpg 

python script/data_prepare.py -s data/room --test_files DSCF4667.JPG DSCF4682.JPG DSCF4794.JPG DSCF4959.JPG \
    --skip_object_mask \
    --text_prompt "sofa. TV. keyboard. mouse. drinking glass. armchair" \
    --groundingdino_checkpoint Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth \
    --groundingdino_config Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py \
    --sam_checkpoint Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth
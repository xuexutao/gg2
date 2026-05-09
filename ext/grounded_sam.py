import argparse
import os
import sys
import copy


import numpy as np
import torch
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


from huggingface_hub import hf_hub_download


def _load_groundingdino_checkpoint_into_model(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state = checkpoint.get('model', checkpoint)
    log = model.load_state_dict(clean_state_dict(state), strict=False)
    print("Model loaded from {} \n => {}".format(ckpt_path, log))
    _ = model.eval()
    return model


def load_model_local(model_config_path: str, model_checkpoint_path: str, device: str = 'cpu'):
    """Load GroundingDINO from local config + checkpoint without HF."""
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device
    return _load_groundingdino_checkpoint_into_model(model, model_checkpoint_path)


def resolve_local_groundingdino_paths(
    local_ckpt_candidates,
    local_cfg_candidates,
):
    """Pick the first existing (ckpt, cfg) pair from candidate lists."""
    for ckpt in local_ckpt_candidates:
        if not ckpt or not os.path.isfile(ckpt):
            continue
        for cfg in local_cfg_candidates:
            if not cfg or not os.path.isfile(cfg):
                continue
            return ckpt, cfg
    return None, None

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return _load_groundingdino_checkpoint_into_model(model, cache_file)

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))




def grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image, BOX_TRESHOLD = 0.3, TEXT_TRESHOLD = 0.45, device='cuda' ):
    image_source = image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(Image.fromarray(image_source), None)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # set image
    sam_predictor.set_image(image_source)
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    if len(boxes_xyxy) > 0:
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )
    else:
        masks = torch.zeros((1,1,H,W)).cuda()

    for i in range(len(masks)):
        annotated_frame_with_mask = show_mask(masks[i][0].cpu().numpy(), annotated_frame)
    
    return torch.sum(masks,dim=0).squeeze().bool(), annotated_frame_with_mask


def select_obj_ioa(
    classification_maps,
    mask,
    ioa_thresh=0.7,
    topk_fallback: int = 3,
):
    """Select predicted object IDs that best match a text mask.

    The upstream heuristic used only IOA (= intersection / class_area) with a high
    threshold (default 0.7). This works for large, compact objects but can fail
    badly for small objects (mouse/keyboard/glass) where the predicted class area
    is much larger than the Grounded-SAM mask → IOA becomes tiny → selects none →
    all-zero prediction masks.

    We keep the original behavior when IOA passes the threshold, but add a robust
    fallback that picks the best matching classes by an F1-like score combining:
      - precision  : IOA  = inter / class_area
      - recall     : inter / mask_area
    This prevents empty selections and makes evaluation stable across object sizes.
    """

    device = classification_maps.device
    mask_f = mask.to(torch.float32)
    mask_area = float(mask_f.sum().item())
    if mask_area <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    # Dynamic minimum intersection to suppress pure-noise matches.
    # Use a low ratio so small objects (mouse/keyboard) can still be selected.
    min_inter = float(max(10, int(mask_area * 0.005)))

    # ---------------------------------------------------------------------
    # Case A: discrete argmax map (H, W)
    # ---------------------------------------------------------------------
    if classification_maps.dim() == 2:
        unique_classes = classification_maps.unique()
        selected = []
        scored = []  # (class_id, f1)

        mask_u8 = (mask_f > 0.5).to(torch.uint8)
        for class_id in unique_classes:
            class_mask = (classification_maps == class_id).to(torch.uint8)
            class_area = float(class_mask.sum().item())
            if class_area <= 0:
                continue

            inter = float((class_mask * mask_u8).sum().item())
            if inter <= 0:
                continue

            ioa = inter / class_area
            recall = inter / mask_area
            f1 = (2.0 * ioa * recall) / (ioa + recall + 1e-6)

            if ioa > ioa_thresh:
                selected.append(int(class_id.item()))
            if inter >= min_inter:
                scored.append((int(class_id.item()), float(f1)))

        if selected:
            return torch.as_tensor(selected, device=device, dtype=torch.long)
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            chosen = [cid for (cid, _score) in scored[: max(1, topk_fallback)]]
            return torch.as_tensor(chosen, device=device, dtype=torch.long)
        return torch.empty((0,), device=device, dtype=torch.long)

    # ---------------------------------------------------------------------
    # Case B: soft probability map (C, H, W)
    #   This is much more robust for small objects where the correct class may
    #   never win argmax on any pixel, yet still has concentrated probability.
    # ---------------------------------------------------------------------
    if classification_maps.dim() == 3:
        prob = classification_maps.to(torch.float32)
        C = prob.shape[0]
        mask_f = mask_f.to(prob.device)

        # expected mass within mask per class (<= mask_area)
        inter = (prob * mask_f.unsqueeze(0)).sum(dim=(1, 2))  # (C,)
        # expected class area (<= H*W)
        class_area = prob.sum(dim=(1, 2)) + 1e-6

        ioa = inter / class_area
        recall = inter / (mask_area + 1e-6)
        f1 = (2.0 * ioa * recall) / (ioa + recall + 1e-6)

        # Only consider classes with meaningful overlap.
        valid = inter >= min_inter
        selected = torch.nonzero((ioa > ioa_thresh) & valid, as_tuple=False).squeeze(-1)
        if selected.numel() > 0:
            return selected.to(device=device, dtype=torch.long)

        # Fallback: take top-k by f1 among valid; if none valid, return empty.
        if valid.any():
            f1_valid = f1.masked_fill(~valid, -1.0)
            k = max(1, topk_fallback)
            _, idx = torch.topk(f1_valid, k=min(k, C), largest=True)
            # filter out the masked -1.0 entries
            idx = idx[f1_valid[idx] > 0]
            return idx.to(device=device, dtype=torch.long)

        return torch.empty((0,), device=device, dtype=torch.long)

    raise ValueError(
        f"select_obj_ioa expects (H,W) argmax map or (C,H,W) prob map, got shape={tuple(classification_maps.shape)}"
    )

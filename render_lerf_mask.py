from __future__ import annotations

# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2
from collections import Counter

from ext.grounded_sam import (
    grouned_sam_output,
    load_model_hf,
    load_model_local,
    resolve_local_groundingdino_paths,
    select_obj_ioa,
)

from segment_anything import sam_model_registry, SamPredictor

from render import feature_to_rgb, visualize_obj


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    classifier,
    groundingdino_model,
    sam_predictor,
    TEXT_PROMPT,
    threshold=0.2,
    debug_prompt_selection=False,
    vote_views: int = 3,
    use_mask_gate: bool = True,
    mask_gate_min_iou: float = 0.2,
    dino_purity_alpha: float = 0.20,
    primary_color_bonus: float = 0.35,
):
    render_path = os.path.join(
        model_path, name, "ours_{}_text".format(iteration), "renders"
    )
    gts_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "gt")
    colormask_path = os.path.join(
        model_path, name, "ours_{}_text".format(iteration), "objects_feature16"
    )
    pred_obj_path = os.path.join(
        model_path, name, "ours_{}_text".format(iteration), "test_mask"
    )
    debug_path = os.path.join(
        model_path, name, "ours_{}_text".format(iteration), "debug_prompt_selection"
    )
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)
    if debug_prompt_selection:
        makedirs(debug_path, exist_ok=True)

    def _filter_by_ref_mask(pred_mask_t: torch.Tensor, ref_mask_t: torch.Tensor):
        """Keep only the connected component best aligned with ref mask.

        This is a *post* filter to suppress "prompt spill" where the classifier
        activates multiple objects (e.g. red chair also covers blue chair).
        We avoid hard intersection (too strict) by selecting the best component.
        """

        if pred_mask_t is None or ref_mask_t is None:
            return pred_mask_t
        if (not bool(pred_mask_t.any().item())) or (not bool(ref_mask_t.any().item())):
            return pred_mask_t

        # If the reference mask is off-target (GroundingDINO mistake), hard
        # gating will *destroy* a correct prediction. So we only gate when the
        # reference overlaps the prediction to a minimum extent.
        overlap = (pred_mask_t & ref_mask_t).sum().float() / (
            pred_mask_t.sum().float() + 1e-6
        )
        if float(overlap.item()) < 0.05:
            return pred_mask_t

        pred_u8 = (pred_mask_t.detach().cpu().numpy().astype(np.uint8))
        ref_u8 = (ref_mask_t.detach().cpu().numpy().astype(np.uint8))

        num, labels = cv2.connectedComponents(pred_u8, connectivity=8)
        if num <= 1:
            return pred_mask_t

        best_idx = -1
        best_score = -1.0
        for i in range(1, num):
            comp = labels == i
            comp_area = float(comp.sum())
            if comp_area < 10:
                continue
            inter = float((comp & (ref_u8 > 0)).sum())
            score = inter / (comp_area + 1e-6)  # precision-like (IOA)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0 or best_score < mask_gate_min_iou:
            return pred_mask_t

        kept = torch.from_numpy((labels == best_idx)).to(pred_mask_t.device)
        return kept.bool()

    def _pick_ref_mask_by_purity(
        prob_map: torch.Tensor,
        masks_all: torch.Tensor,
        dino_scores=None,
        image_rgb: np.ndarray | None = None,
        prompt: str | None = None,
    ):
        """Pick the candidate Grounded-SAM mask that best matches a *single* classifier class.

        This uses the classifier prob_map as a judge to avoid GroundingDINO picking
        the wrong instance when multiple boxes exist.

        Args:
            prob_map: (C,H,W) softmax over classes
            masks_all: (B,H,W) bool masks from SAM
            dino_scores: optional (B,) tensor/cpu list; used as a weak prior
        """
        if masks_all is None or masks_all.numel() == 0:
            return None
        if masks_all.dim() != 3:
            return None

        device = prob_map.device
        masks_f = masks_all.to(device=device, dtype=torch.float32)  # (B,H,W)
        areas = masks_f.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)

        # class_mass[b,c] = sum_{pixels in mask_b} p_c
        class_mass = (prob_map.unsqueeze(0) * masks_f.unsqueeze(1)).sum(dim=(2, 3))  # (B,C)
        best_mass, _ = class_mass.max(dim=1)  # (B,)
        purity = best_mass / areas  # (B,) in [0,1]

        score = purity
        if dino_scores is not None:
            ds = torch.as_tensor(dino_scores, device=device, dtype=torch.float32)
            if ds.numel() == score.numel():
                # normalize to [0,1] (roughly) then add as weak prior
                ds = (ds - ds.min()) / (ds.max() - ds.min() + 1e-6)
                score = score + float(dino_purity_alpha) * ds

        # If the prompt is of the form "<color> <noun>" (e.g. red apple),
        # use a cheap color-dominance heuristic to disambiguate instances.
        if (
            image_rgb is not None
            and prompt is not None
            and isinstance(prompt, str)
            and len(prompt.split()) > 0
        ):
            first = prompt.split()[0].lower()
            if first in {"red", "green", "blue", "yellow", "white", "black", "brown", "orange", "pink", "purple"}:
                img = image_rgb

                def _color_score_local(mask_bool: np.ndarray, color: str) -> float:
                    if mask_bool.sum() < 10:
                        return 0.0
                    region = img[mask_bool].astype(np.float32) / 255.0
                    r = float(region[:, 0].mean())
                    g = float(region[:, 1].mean())
                    b = float(region[:, 2].mean())
                    if color == "red":
                        s = r - 0.5 * (g + b)
                    elif color == "green":
                        s = g - 0.5 * (r + b)
                    elif color == "blue":
                        s = b - 0.5 * (r + g)
                    elif color == "yellow":
                        s = 0.5 * (r + g) - b
                    elif color == "white":
                        s = (r + g + b) / 3.0 - 0.5
                    elif color == "black":
                        s = 0.5 - (r + g + b) / 3.0
                    else:
                        s = 0.0
                    return float(np.clip(s, 0.0, 1.0))

                bonus = []
                masks_np = masks_all.detach().cpu().numpy().astype(bool)
                for bi in range(masks_np.shape[0]):
                    bonus.append(_color_score_local(masks_np[bi], first))
                bonus_t = torch.as_tensor(bonus, device=device, dtype=torch.float32)
                score = score + float(primary_color_bonus) * bonus_t

        best_i = int(torch.argmax(score).item())
        return masks_all[best_i].to(device=device).bool()

    # ---------------------------------------------------------------------
    # Robust prompt→object-id matching
    #   - Grounded-SAM on multiple views (vote)
    #   - pick stable object IDs across views
    # ---------------------------------------------------------------------
    vote_views_eff = max(1, min(int(vote_views), len(views)))
    voted_ids = []
    first_annotated = None
    first_text_mask0 = None
    first_prob0 = None

    for vi in range(vote_views_eff):
        results0 = render(views[vi], gaussians, pipeline, background)
        rendering0 = results0["render"]
        rendering_obj0 = results0["render_object"]
        logits0 = classifier(rendering_obj0)
        prob0 = torch.softmax(logits0, dim=0)

        image0 = (rendering0.permute(1, 2, 0) * 255).cpu().numpy().astype("uint8")
        text_mask0, annotated0, masks_all0, dino_scores0 = grouned_sam_output(
            groundingdino_model,
            sam_predictor,
            TEXT_PROMPT,
            image0,
            mask_mode="best",
            return_all=True,
        )

        # Replace the DINO-chosen mask by a classifier-consistent one.
        ref0 = _pick_ref_mask_by_purity(
            prob0,
            masks_all0,
            dino_scores0,
            image_rgb=image0,
            prompt=TEXT_PROMPT,
        )
        if ref0 is not None:
            text_mask0 = ref0
        if vi == 0:
            first_annotated = annotated0
            first_text_mask0 = text_mask0
            first_prob0 = prob0

        ids0 = select_obj_ioa(prob0, text_mask0)
        voted_ids.extend([int(x.item()) for x in ids0])

    if first_annotated is not None:
        Image.fromarray(first_annotated).save(
            os.path.join(render_path[:-8], "grounded-sam---" + TEXT_PROMPT + ".png")
        )

    if voted_ids:
        cnt = Counter(voted_ids)
        # keep IDs that appear in >= half of the vote views; fallback to top-1.
        keep = [cid for cid, c in cnt.items() if c >= (vote_views_eff + 1) // 2]
        if not keep:
            keep = [cnt.most_common(1)[0][0]]
        selected_obj_ids = torch.as_tensor(keep, device=gaussians._xyz.device, dtype=torch.long)
    else:
        selected_obj_ids = torch.empty((0,), device=gaussians._xyz.device, dtype=torch.long)

    if debug_prompt_selection and (first_text_mask0 is not None) and (first_prob0 is not None):
        # Report top classes by *soft* IOA for easier debugging.
        text_mask_f = first_text_mask0.to(torch.float32)
        inter = (first_prob0 * text_mask_f.unsqueeze(0)).sum(dim=(1, 2))
        class_area = first_prob0.sum(dim=(1, 2)) + 1e-6
        ioa = (inter / class_area).detach().cpu().numpy()
        inter_np = inter.detach().cpu().numpy()
        class_area_np = class_area.detach().cpu().numpy()

        debug_rows = []
        for cid in range(first_prob0.shape[0]):
            debug_rows.append(
                (
                    int(cid),
                    float(ioa[cid]),
                    float(class_area_np[cid]),
                    float(inter_np[cid]),
                )
            )
        debug_rows.sort(key=lambda x: x[1], reverse=True)
        debug_file = os.path.join(debug_path, TEXT_PROMPT + ".txt")
        with open(debug_file, "w") as f:
            f.write(f"prompt={TEXT_PROMPT}\n")
            f.write(f"selected_obj_ids={[int(x.item()) for x in selected_obj_ids]}\n")
            f.write(f"text_mask_pixels={int(text_mask_f.sum().item())}\n")
            f.write("top_ioa_classes:\n")
            for class_id, ioa, class_area, inter in debug_rows[:20]:
                f.write(
                    f"  class_id={class_id} ioa={ioa:.6f} class_area={class_area} intersection={inter}\n"
                )

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pred_obj_img_path = os.path.join(pred_obj_path, str(idx))
        makedirs(pred_obj_img_path, exist_ok=True)

        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)

        # Optional per-view Grounded-SAM reference mask (for gating).
        ref_mask_t = None
        if use_mask_gate:
            image_ref = (rendering.permute(1, 2, 0) * 255).detach().cpu().numpy().astype("uint8")
            # Use classifier to select the best candidate mask for this view.
            prob_ref = torch.softmax(logits, dim=0)
            ref_mask_t, _, masks_all_ref, dino_scores_ref = grouned_sam_output(
                groundingdino_model,
                sam_predictor,
                TEXT_PROMPT,
                image_ref,
                mask_mode="best",
                return_all=True,
            )
            picked = _pick_ref_mask_by_purity(
                prob_ref,
                masks_all_ref,
                dino_scores_ref,
                image_rgb=image_ref,
                prompt=TEXT_PROMPT,
            )
            if picked is not None:
                ref_mask_t = picked

        if len(selected_obj_ids) > 0:
            prob = torch.softmax(logits, dim=0)

            # Thresholding can easily erase small objects (mouse/glass) even when
            # the correct IDs are selected. Use a simple adaptive fallback.
            def build_mask(th: float):
                m = prob[selected_obj_ids, :, :] > th
                return m.any(dim=0)

            pred_obj_mask_t = build_mask(threshold)
            if not bool(pred_obj_mask_t.any().item()):
                for th in [threshold * 0.5, threshold * 0.25]:
                    pred_obj_mask_t = build_mask(th)
                    if bool(pred_obj_mask_t.any().item()):
                        break

            if use_mask_gate and (ref_mask_t is not None):
                pred_obj_mask_t = _filter_by_ref_mask(pred_obj_mask_t, ref_mask_t)

            pred_obj_mask = (pred_obj_mask_t.squeeze().cpu().numpy() * 255).astype(
                np.uint8
            )
        else:
            pred_obj_mask = (
                torch.zeros_like(view.objects, dtype=torch.uint8).cpu().numpy()
            )

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(
            os.path.join(colormask_path, "{0:05d}".format(idx) + ".png")
        )
        Image.fromarray(pred_obj_mask).save(
            os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png")
        )
        print(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    debug_prompt_selection: bool,
):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        num_classes = dataset.num_classes
        print("Num classes: ", num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(
            torch.load(
                os.path.join(
                    dataset.model_path,
                    "point_cloud",
                    "iteration_" + str(scene.loaded_iter),
                    "classifier.pth",
                )
            )
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        print("Loading Grounding DINO model...")

        # Prefer local checkpoints in Tracking-Anything-with-DEVA/saves/.
        # Fallback to HuggingFace if missing.
        local_ckpt_candidates = [
            os.path.join(
                "Tracking-Anything-with-DEVA", "saves", "groundingdino_swint_ogc.pth"
            ),
        ]
        local_cfg_candidates = [
            os.path.join(
                "Tracking-Anything-with-DEVA", "saves", "GroundingDINO_SwinT_OGC.py"
            ),
        ]

        local_ckpt, local_cfg = resolve_local_groundingdino_paths(
            local_ckpt_candidates, local_cfg_candidates
        )
        if local_ckpt and local_cfg:
            print(f"Using local GroundingDINO: ckpt={local_ckpt}, cfg={local_cfg}")
            groundingdino_model = load_model_local(local_cfg, local_ckpt)
        else:
            ckpt_repo_id = "ShilongLiu/GroundingDINO"
            ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
            ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
            print("Local GroundingDINO not found, fallback to HuggingFace download...")
            groundingdino_model = load_model_hf(
                ckpt_repo_id, ckpt_filenmae, ckpt_config_filename
            )
        print("Grounding DINO model loaded.")

        # sam-hq
        print("Loading SAM-HQ model...")
        sam_checkpoint = "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        sam_predictor = SamPredictor(sam)
        print("SAM-HQ model loaded.")

        # Text prompt
        if "figurines" in dataset.model_path:
            positive_input = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat"
        elif "ramen" in dataset.model_path:
            positive_input = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl"
        elif "teatime" in dataset.model_path:
            positive_input = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass"
        elif "room" in dataset.model_path:
            positive_input = "sofa;TV;keyboard;mouse;drinking glass;armchair"
        else:
            raise NotImplementedError  # You can provide your text prompt here

        positives = positive_input.split(";")
        print("Text prompts:    ", positives)

        for TEXT_PROMPT in positives:
            if not skip_train:
                render_set(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    scene.getTrainCameras(),
                    gaussians,
                    pipeline,
                    background,
                    classifier,
                    groundingdino_model,
                    sam_predictor,
                    TEXT_PROMPT,
                    debug_prompt_selection=debug_prompt_selection,
                )
            if not skip_test:
                render_set(
                    dataset.model_path,
                    "test",
                    scene.loaded_iter,
                    scene.getTestCameras(),
                    gaussians,
                    pipeline,
                    background,
                    classifier,
                    groundingdino_model,
                    sam_predictor,
                    TEXT_PROMPT,
                    debug_prompt_selection=debug_prompt_selection,
                )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--debug_prompt_selection", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.debug_prompt_selection,
    )

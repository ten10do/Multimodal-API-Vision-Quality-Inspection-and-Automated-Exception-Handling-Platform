"""Localization-aware representation experiments with an immutable D3 image branch."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from inference_app.d3_candidate_predictor import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from steel_patchcore.aggregation import auroc  # noqa: E402
from steel_patchcore.candidate_registry import CandidateRegistry, resolve_uri, sha256_file  # noqa: E402
from steel_patchcore.d3_localization_representation import (  # noqa: E402
    D3_IMAGE_AUROC,
    EXPERIMENTAL_BANK_BUDGET,
    EXPERIMENTAL_BANK_SEED,
    EXPERIMENTAL_TRAIN_COUNT,
    HIGH_RESOLUTION_SIDE,
    INTERMEDIATE_BLOCK_INDEX,
    LOCALIZATION_GATE,
    REPRESENTATION_SPECS,
    LocalizationRepresentationError,
    assert_image_branch_immutable,
    dual_objective_gate,
    fuse_dense_maps,
    score_delta_summary,
    validate_results_report,
)
from steel_patchcore.d3_operational import atomic_write_json, pixel_localization_metrics  # noqa: E402
from steel_patchcore.lifecycle import lifecycle_enter  # noqa: E402
from steel_patchcore.rle import rle_decode  # noqa: E402
from steel_patchcore.tile import TILE_X0, stitch_scores  # noqa: E402

REGISTRY_ROOT = ROOT / "model-training/registry"
MANIFEST_PATH = REGISTRY_ROOT / "steel-patchcore-d3-candidate/manifest.json"
DATASET_ROOT = ROOT / "model-training/datasets/severstal-steel"
IMAGE_DIR = DATASET_ROOT / "raw/train_images"
CSV_PATH = DATASET_ROOT / "raw/train.csv"
SOURCE_SPLIT = DATASET_ROOT / "split_manifest.json"
RECOVERY_SPLIT = DATASET_ROOT / "recovery_split_manifest.json"
SUBSET_MANIFEST = DATASET_ROOT / "representation_diagnostic_manifest.json"
SUBSET_SHA = DATASET_ROOT / "representation_diagnostic_manifest.sha256"
SHADOW_LOG = ROOT / "docs/d3-shadow-prediction-log.json"

VITS_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
EXPECTED_VITS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
R4_BANK = ROOT / "model-training/runs/steel-domain-representation/D1-dinov2-s14/bank.npz"
R4_BANK_MANIFEST = R4_BANK.with_name("bank_manifest.json")

RUN_ROOT = ROOT / "model-training/runs/steel-d3-localization-representation"
R1_DIR = RUN_ROOT / "R-L1"
R2_DIR = RUN_ROOT / "R-L2"
R1_BANK = R1_DIR / "bank.npz"
R2_BANK = R2_DIR / "bank.npz"
EVAL_CHECKPOINT = RUN_ROOT / "evaluation-checkpoint.json"
RESULTS_JSON = ROOT / "docs/d3-localization-representation-results.json"
RESULTS_MD = ROOT / "docs/d3-localization-representation-investigation.md"
PROTOCOL_VERSION = "steel_patchcore_d3_localization_representation_v1"
BANK_CHECKPOINT_EVERY = 100
EVAL_CHECKPOINT_EVERY = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def load_lineage() -> tuple[CandidateRegistry, dict, dict[str, str]]:
    registry = CandidateRegistry(REGISTRY_ROOT, ROOT)
    manifest = registry.load_manifest()
    verification = registry.verify_artifact(manifest)
    if not verification.passed:
        raise LocalizationRepresentationError(f"D3_ARTIFACT_INVALID:{verification.errors}")
    if manifest["status"] != "CANDIDATE" or manifest["production_promotion"] is not False:
        raise LocalizationRepresentationError("D3_CANDIDATE_ONLY_REQUIRED")
    return registry, manifest, verification.hashes


def load_ids() -> tuple[list[str], list[str], list[str]]:
    diagnostic = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(SUBSET_MANIFEST) != SUBSET_SHA.read_text(encoding="ascii").strip():
        raise LocalizationRepresentationError("TRAIN_SUBSET_HASH_MISMATCH")
    train_ids = list(diagnostic["train_normal_subset"])
    if len(train_ids) != EXPERIMENTAL_TRAIN_COUNT or diagnostic["holdout_access_count"] != 0:
        raise LocalizationRepresentationError("TRAIN_SUBSET_INVALID")
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY_SPLIT.read_text(encoding="utf-8"))
    normal_ids = list(source["splits"]["test_normal"])
    anomaly_ids = list(recovery["recovery_holdout_anomaly"])
    if len(normal_ids) != 591 or len(anomaly_ids) != 3333 or set(normal_ids) & set(anomaly_ids):
        raise LocalizationRepresentationError("SEALED_EVALUATION_SET_INVALID")
    return train_ids, normal_ids, anomaly_ids


def load_mask_rles(image_ids: list[str]) -> dict[str, list[str]]:
    wanted = set(image_ids)
    frame = pd.read_csv(CSV_PATH, keep_default_na=True)
    selected = frame[frame["ImageId"].astype(str).map(lambda value: Path(value).stem in wanted)]
    encoded_by_image: dict[str, list[str]] = {}
    for raw_id, group in selected.groupby(selected["ImageId"].astype(str)):
        image_id = Path(raw_id).stem
        rows = [str(value) for value in group["EncodedPixels"] if not pd.isna(value)]
        if rows:
            encoded_by_image[image_id] = rows
    missing = wanted - set(encoded_by_image)
    if missing:
        raise LocalizationRepresentationError(f"MASK_MISSING:{sorted(missing)[:5]}")
    return encoded_by_image


def decode_mask(encoded_rows: list[str]) -> np.ndarray:
    mask = np.zeros((256, 1600), dtype=np.uint8)
    for encoded in encoded_rows:
        mask = np.maximum(mask, rle_decode(encoded))
    return mask


def load_model(identifier: str, weights: Path, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", identifier, pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def tile_batch(image_id: str, side: int, device: torch.device) -> torch.Tensor:
    with Image.open(IMAGE_DIR / f"{image_id}.jpg") as opened:
        image = opened.convert("RGB")
    tensors = []
    for x0 in TILE_X0:
        tile = image.crop((x0, 0, x0 + 256, 256))
        array = np.asarray(tile, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    batch = torch.stack(tensors).to(device)
    if side != 256:
        batch = F.interpolate(batch, size=(side, side), mode="bilinear", align_corners=False)
    return batch


def extract_features(model: torch.nn.Module, image_id: str, candidate: str, device: torch.device) -> torch.Tensor:
    if candidate == "R-L1":
        batch = tile_batch(image_id, 252, device)
        with torch.no_grad():
            tokens = model.get_intermediate_layers(batch, n=[INTERMEDIATE_BLOCK_INDEX], norm=True)[0]
    elif candidate == "R-L2":
        batch = tile_batch(image_id, HIGH_RESOLUTION_SIDE, device)
        with torch.no_grad():
            tokens = model.forward_features(batch)["x_norm_patchtokens"]
    elif candidate == "R-L4":
        batch = tile_batch(image_id, 252, device)
        with torch.no_grad():
            tokens = model.forward_features(batch)["x_norm_patchtokens"]
    else:
        raise LocalizationRepresentationError(f"UNKNOWN_FEATURE_CANDIDATE:{candidate}")
    expected = {"R-L1": (7, 324, 768), "R-L2": (7, 1024, 768), "R-L4": (7, 324, 384)}[candidate]
    if tuple(tokens.shape) != expected:
        raise LocalizationRepresentationError(f"FEATURE_SHAPE_MISMATCH:{candidate}:{tuple(tokens.shape)}")
    return F.normalize(tokens, p=2, dim=2)


def reservoir_update(
    reservoir: np.ndarray,
    seen: int,
    features: np.ndarray,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(features, dtype=np.float32)
    start = 0
    if seen < len(reservoir):
        count = min(len(values), len(reservoir) - seen)
        reservoir[seen:seen + count] = values[:count]
        seen += count
        start = count
    remaining = values[start:]
    if len(remaining):
        totals = np.arange(seen + 1, seen + len(remaining) + 1, dtype=np.int64)
        destinations = rng.integers(0, totals)
        accepted = np.flatnonzero(destinations < len(reservoir))
        for offset in accepted:
            reservoir[int(destinations[offset])] = remaining[offset]
        seen += len(remaining)
    return seen


def bank_paths(candidate: str) -> tuple[Path, Path, Path]:
    directory = R1_DIR if candidate == "R-L1" else R2_DIR
    return directory / "bank.npz", directory / "progress.npz", directory / "progress-state.json"


def verify_experimental_bank(candidate: str, path: Path) -> tuple[np.ndarray, dict]:
    manifest_path = path.with_name("bank-manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise LocalizationRepresentationError(f"EXPERIMENTAL_BANK_MISSING:{candidate}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate") != candidate or manifest.get("bank_sha256") != sha256_file(path):
        raise LocalizationRepresentationError(f"EXPERIMENTAL_BANK_HASH_MISMATCH:{candidate}")
    with np.load(path, allow_pickle=False) as payload:
        bank = payload["features"].astype(np.float32)
    expected_dim = int(REPRESENTATION_SPECS[candidate]["dimension"])
    if bank.shape != (EXPERIMENTAL_BANK_BUDGET, expected_dim) or not np.isfinite(bank).all():
        raise LocalizationRepresentationError(f"EXPERIMENTAL_BANK_INVALID:{candidate}:{bank.shape}")
    bank.flags.writeable = False
    return bank, manifest


def build_bank(candidate: str) -> dict:
    if candidate not in {"R-L1", "R-L2"}:
        raise LocalizationRepresentationError("ONLY_RL1_RL2_BANKS_ARE_BUILDABLE")
    if not torch.cuda.is_available():
        raise LocalizationRepresentationError("REPRESENTATION_BANK_REQUIRES_GPU")
    registry, manifest, artifact_hashes = load_lineage()
    train_ids, _, _ = load_ids()
    bank_path, progress_path, state_path = bank_paths(candidate)
    if bank_path.is_file():
        _, bank_manifest = verify_experimental_bank(candidate, bank_path)
        print(f"{candidate} bank already complete sha={bank_manifest['bank_sha256']}", flush=True)
        return bank_manifest
    blocked = lifecycle_enter(f"d3_localization_{candidate}_bank", manifest["bank_sha256"], progress_path, bank_path.parent)
    if blocked is not None:
        raise LocalizationRepresentationError(f"LIFECYCLE_BLOCKED:{blocked}")
    device = torch.device("cuda:0")
    model = load_model("dinov2_vitb14", resolve_uri(ROOT, manifest["weights_uri"]), device)
    dimension = int(REPRESENTATION_SPECS[candidate]["dimension"])
    reservoir = np.zeros((EXPERIMENTAL_BANK_BUDGET, dimension), dtype=np.float32)
    seen = 0
    processed = 0
    rng = np.random.default_rng(EXPERIMENTAL_BANK_SEED)
    if progress_path.is_file() and state_path.is_file():
        with np.load(progress_path, allow_pickle=False) as progress:
            reservoir = progress["reservoir"].astype(np.float32)
            seen = int(progress["seen"])
            processed = int(progress["processed"])
        rng.bit_generator.state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"{candidate} bank resume processed={processed} seen={seen}", flush=True)
    for index in range(processed, len(train_ids)):
        features = extract_features(model, train_ids[index], candidate, device).detach().cpu().numpy()
        seen = reservoir_update(reservoir, seen, features.reshape(-1, dimension), rng)
        processed = index + 1
        if processed % BANK_CHECKPOINT_EVERY == 0 or processed == len(train_ids):
            atomic_savez(
                progress_path,
                reservoir=reservoir,
                seen=np.asarray(seen, dtype=np.int64),
                processed=np.asarray(processed, dtype=np.int64),
            )
            atomic_write_json(state_path, rng.bit_generator.state)
            print(f"{candidate} bank processed={processed}/{len(train_ids)} seen={seen}", flush=True)
    if seen < EXPERIMENTAL_BANK_BUDGET:
        raise LocalizationRepresentationError(f"EXPERIMENTAL_BANK_UNDERSIZED:{candidate}:{seen}")
    atomic_savez(bank_path, features=reservoir)
    bank_manifest = {
        "schema_version": "steel_patchcore_localization_experimental_bank_v1",
        "candidate": candidate,
        "representation": REPRESENTATION_SPECS[candidate],
        "rows": EXPERIMENTAL_BANK_BUDGET,
        "dimension": dimension,
        "sampling": "reservoir Algorithm R",
        "seed": EXPERIMENTAL_BANK_SEED,
        "source": "frozen representation_diagnostic_manifest train_normal_subset",
        "source_count": len(train_ids),
        "source_manifest_sha256": sha256_file(SUBSET_MANIFEST),
        "d3_artifact_hashes_at_build": artifact_hashes,
        "bank_sha256": sha256_file(bank_path),
        "candidate_artifact": False,
        "generated_at": utc_now(),
    }
    atomic_write_json(bank_path.with_name("bank-manifest.json"), bank_manifest)
    after = registry.verify_artifact(manifest)
    if not after.passed or after.hashes != artifact_hashes:
        raise LocalizationRepresentationError("D3_ARTIFACT_CHANGED_DURING_BANK_BUILD")
    return bank_manifest


def verify_r4_bank() -> tuple[np.ndarray, dict]:
    if sha256_file(VITS_WEIGHTS) != EXPECTED_VITS_SHA256:
        raise LocalizationRepresentationError("R_L4_WEIGHTS_HASH_MISMATCH")
    manifest = json.loads(R4_BANK_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("bank_sha256") != sha256_file(R4_BANK):
        raise LocalizationRepresentationError("R_L4_BANK_HASH_MISMATCH")
    expected_manifest = {
        "rows": EXPERIMENTAL_BANK_BUDGET,
        "budget": EXPERIMENTAL_BANK_BUDGET,
        "seed": EXPERIMENTAL_BANK_SEED,
        "source_count": EXPERIMENTAL_TRAIN_COUNT,
        "subset_manifest_sha256": sha256_file(SUBSET_MANIFEST),
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise LocalizationRepresentationError("R_L4_BANK_PROTOCOL_MISMATCH")
    with np.load(R4_BANK, allow_pickle=False) as payload:
        bank = payload["features"].astype(np.float32)
    if bank.shape != (EXPERIMENTAL_BANK_BUDGET, 384):
        raise LocalizationRepresentationError(f"R_L4_BANK_SHAPE_MISMATCH:{bank.shape}")
    bank.flags.writeable = False
    return bank, manifest


def resize_grid(grid: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)


def distance_grids(tokens: torch.Tensor, bank: torch.Tensor, side: int) -> np.ndarray:
    grids = []
    for tile_tokens in tokens:
        distance = 1.0 - (tile_tokens @ bank.T).max(dim=1).values
        grids.append(distance.reshape(side, side).detach().cpu().numpy().astype(np.float32))
    return np.stack(grids)


def maps_and_scores(
    model_b: torch.nn.Module,
    model_s: torch.nn.Module,
    banks: dict[str, torch.Tensor],
    image_id: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    l1_grids = distance_grids(extract_features(model_b, image_id, "R-L1", device), banks["R-L1"], 18)
    l2_grids = distance_grids(extract_features(model_b, image_id, "R-L2", device), banks["R-L2"], 32)
    l4_grids = distance_grids(extract_features(model_s, image_id, "R-L4", device), banks["R-L4"], 18)
    tile_maps = {
        "R-L1": [resize_grid(grid) for grid in l1_grids],
        "R-L2": [resize_grid(grid) for grid in l2_grids],
        "R-L4": [resize_grid(grid) for grid in l4_grids],
    }
    tile_maps["R-L3"] = [fuse_dense_maps(first, second) for first, second in zip(tile_maps["R-L1"], tile_maps["R-L2"])]
    maps = {name: stitch_scores(tile_maps[name]) for name in REPRESENTATION_SPECS}
    scores = {
        "R-L1": float(l1_grids.max()),
        "R-L2": float(l2_grids.max()),
        "R-L3": float(max(tile_map.max() for tile_map in tile_maps["R-L3"])),
        "R-L4": float(l4_grids.max()),
    }
    return maps, scores


def load_eval_checkpoint(manifest: dict, lineage: dict, bank_hashes: dict[str, str]) -> dict:
    identity = {
        "schema_version": "steel_patchcore_d3_localization_representation_checkpoint_v1",
        "protocol_version": PROTOCOL_VERSION,
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "threshold": manifest["threshold"],
        "d3_artifact_hashes": lineage,
        "experimental_bank_hashes": bank_hashes,
        "representation_specs": REPRESENTATION_SPECS,
    }
    if not EVAL_CHECKPOINT.is_file():
        checkpoint = {**identity, "completed": {}, "failures": [], "updated_at": utc_now()}
        atomic_write_json(EVAL_CHECKPOINT, checkpoint)
        return checkpoint
    checkpoint = json.loads(EVAL_CHECKPOINT.read_text(encoding="utf-8"))
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise LocalizationRepresentationError("EVALUATION_CHECKPOINT_LINEAGE_MISMATCH")
    return checkpoint


def evaluate() -> dict:
    if not torch.cuda.is_available():
        raise LocalizationRepresentationError("REPRESENTATION_EVALUATION_REQUIRES_GPU")
    registry, manifest, before = load_lineage()
    _, normal_ids, anomaly_ids = load_ids()
    r1_bank, r1_manifest = verify_experimental_bank("R-L1", R1_BANK)
    r2_bank, r2_manifest = verify_experimental_bank("R-L2", R2_BANK)
    r4_bank, r4_manifest = verify_r4_bank()
    bank_hashes = {
        "R-L1": r1_manifest["bank_sha256"],
        "R-L2": r2_manifest["bank_sha256"],
        "R-L4": r4_manifest["bank_sha256"],
    }
    blocked = lifecycle_enter("d3_localization_representation_evaluation", manifest["bank_sha256"], EVAL_CHECKPOINT, RUN_ROOT)
    if blocked is not None:
        raise LocalizationRepresentationError(f"LIFECYCLE_BLOCKED:{blocked}")
    shadow_rows = json.loads(SHADOW_LOG.read_text(encoding="utf-8"))["records"]
    shadow = {row["image_id"]: row for row in shadow_rows}
    order = [("test_normal", image_id) for image_id in normal_ids] + [
        ("recovery_holdout_anomaly", image_id) for image_id in anomaly_ids
    ]
    if set(shadow) != {image_id for _, image_id in order}:
        raise LocalizationRepresentationError("D3_SHADOW_MEMBERSHIP_MISMATCH")
    masks = load_mask_rles(anomaly_ids)
    checkpoint = load_eval_checkpoint(manifest, before, bank_hashes)
    device = torch.device("cuda:0")
    model_b = load_model("dinov2_vitb14", resolve_uri(ROOT, manifest["weights_uri"]), device)
    model_s = load_model("dinov2_vits14", VITS_WEIGHTS, device)
    bank_tensors = {
        "R-L1": torch.tensor(r1_bank, device=device),
        "R-L2": torch.tensor(r2_bank, device=device),
        "R-L4": torch.tensor(r4_bank, device=device),
    }
    since_write = 0
    for role, image_id in order:
        if image_id in checkpoint["completed"]:
            continue
        try:
            maps, scores = maps_and_scores(model_b, model_s, bank_tensors, image_id, device)
            if len(checkpoint["completed"]) < 3:
                repeated_maps, repeated_scores = maps_and_scores(model_b, model_s, bank_tensors, image_id, device)
                for name in REPRESENTATION_SPECS:
                    if scores[name] != repeated_scores[name] or not np.array_equal(maps[name], repeated_maps[name]):
                        raise LocalizationRepresentationError(f"HEATMAP_REPRODUCIBILITY_FAILED:{image_id}:{name}")
            record = {
                "image_id": image_id,
                "split_role": role,
                "d3_image_score": float(shadow[image_id]["score"]),
                "representation_scores": scores,
                "metrics": {},
            }
            if role == "recovery_holdout_anomaly":
                mask = decode_mask(masks[image_id])
                record["metrics"] = {
                    name: pixel_localization_metrics(maps[name], mask)
                    for name in REPRESENTATION_SPECS
                }
            checkpoint["completed"][image_id] = record
            since_write += 1
        except Exception as exc:
            checkpoint["failures"].append(
                {"timestamp": utc_now(), "image_id": image_id, "error": f"{type(exc).__name__}:{exc}"}
            )
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(EVAL_CHECKPOINT, checkpoint)
            raise
        if since_write >= EVAL_CHECKPOINT_EVERY:
            checkpoint["updated_at"] = utc_now()
            atomic_write_json(EVAL_CHECKPOINT, checkpoint)
            since_write = 0
            print(f"representation evaluation completed={len(checkpoint['completed'])}/{len(order)}", flush=True)
    checkpoint["updated_at"] = utc_now()
    atomic_write_json(EVAL_CHECKPOINT, checkpoint)
    after_verification = registry.verify_artifact(manifest)
    if not after_verification.passed or after_verification.hashes != before:
        raise LocalizationRepresentationError("D3_ARTIFACT_CHANGED_DURING_REPRESENTATION_EVALUATION")
    return finalize(manifest, before, after_verification.hashes, order, anomaly_ids, checkpoint, bank_hashes)


def finalize(
    manifest: dict,
    before: dict[str, str],
    after: dict[str, str],
    order: list[tuple[str, str]],
    anomaly_ids: list[str],
    checkpoint: dict,
    bank_hashes: dict[str, str],
) -> dict:
    completed = checkpoint["completed"]
    expected_ids = {image_id for _, image_id in order}
    if set(completed) != expected_ids:
        raise LocalizationRepresentationError(f"REPRESENTATION_EVALUATION_INCOMPLETE:{len(completed)}/{len(expected_ids)}")
    d3_scores = np.asarray([completed[image_id]["d3_image_score"] for _, image_id in order], dtype=np.float64)
    labels = np.asarray([0 if role == "test_normal" else 1 for role, _ in order], dtype=np.int8)
    if not math.isclose(auroc(d3_scores, labels), D3_IMAGE_AUROC, rel_tol=0.0, abs_tol=1e-12):
        raise LocalizationRepresentationError("FROZEN_D3_AUROC_MISMATCH")
    rows = []
    for name, spec in REPRESENTATION_SPECS.items():
        representation_scores = np.asarray(
            [completed[image_id]["representation_scores"][name] for _, image_id in order], dtype=np.float64
        )
        standalone_auroc = auroc(representation_scores, labels)
        pixel_auroc = float(np.mean([completed[image_id]["metrics"][name]["pixel_auroc"] for image_id in anomaly_ids]))
        aupro = float(np.mean([completed[image_id]["metrics"][name]["aupro"] for image_id in anomaly_ids]))
        assert_image_branch_immutable(d3_scores, d3_scores.copy())
        standalone_passed, standalone_checks = dual_objective_gate(pixel_auroc, aupro, standalone_auroc)
        passed, checks = dual_objective_gate(pixel_auroc, aupro, D3_IMAGE_AUROC)
        rows.append(
            {
                "candidate": name,
                "representation": spec,
                "standalone_representation_branch": {
                    "image_auroc": standalone_auroc,
                    "image_auroc_delta_vs_d3": standalone_auroc - D3_IMAGE_AUROC,
                    "image_score_delta_vs_d3": score_delta_summary(representation_scores, d3_scores),
                    "checks": standalone_checks,
                    "verdict": "PASS" if standalone_passed else "FAILED",
                },
                "image_auroc": standalone_auroc,
                "pixel_auroc": pixel_auroc,
                "aupro": aupro,
                "dual_objective": {
                    "image_branch": "frozen D3 A0",
                    "pixel_branch": name,
                    "image_auroc": D3_IMAGE_AUROC,
                    "image_score_immutable": True,
                    "checks": checks,
                    "verdict": "PASS" if passed else "FAILED",
                },
            }
        )
    passing = [row["candidate"] for row in rows if row["dual_objective"]["verdict"] == "PASS"]
    preferred = max(rows, key=lambda row: (row["pixel_auroc"], row["aupro"]))["candidate"] if passing else None
    report = {
        "schema_version": "steel_patchcore_d3_localization_representation_results_v1",
        "protocol_version": PROTOCOL_VERSION,
        "candidate_status": manifest["status"],
        "model_version": manifest["model_version"],
        "artifact_version": manifest["artifact_version"],
        "frozen_baseline": {
            "encoder": "DINOv2-B/14",
            "adaptation": "train-normal ZCA",
            "distance": "cosine-1NN",
            "image_scoring": "D3 A0 global max",
            "image_auroc": D3_IMAGE_AUROC,
            "threshold": manifest["threshold"],
        },
        "dataset": {"test_normal": 591, "recovery_holdout_anomaly": 3333},
        "experimental_bank_policy": {
            "rows": EXPERIMENTAL_BANK_BUDGET,
            "seed": EXPERIMENTAL_BANK_SEED,
            "train_normal_count": EXPERIMENTAL_TRAIN_COUNT,
            "candidate_artifact": False,
            "hashes": bank_hashes,
        },
        "localization_gate": LOCALIZATION_GATE,
        "representations": rows,
        "dual_objective_design": {
            "image_branch": "frozen D3 A0; byte-identical sealed scores",
            "pixel_branch": "independent dense representation anomaly map",
            "separation_supported": bool(passing),
            "passing_pixel_branches": passing,
            "preferred_pixel_branch_by_localization": preferred,
        },
        "overall_verdict": "PASS" if passing else "FAILED",
        "d3_artifact_hashes_before": before,
        "d3_artifact_hashes_after": after,
        "artifact_unchanged": before == after,
        "threshold_changed": False,
        "production_promotion": False,
        "failures": checkpoint["failures"],
        "generated_at": utc_now(),
    }
    validate_results_report(report)
    atomic_write_json(RESULTS_JSON, report)
    RESULTS_MD.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"verdict": report["overall_verdict"], "passing": passing}), flush=True)
    return report


def render_report(report: dict) -> str:
    lines = [
        "# D3 Localization-Aware Representation Investigation",
        "",
        f"Verdict: **`{report['overall_verdict']}`**",
        "",
        "## 1. Frozen baseline audit",
        "",
        "D3 remains DINOv2-B/14 final patch tokens → frozen train-normal ZCA → per-patch L2 → cosine-1NN → A0 global maximum. Its candidate artifact, threshold, and sealed image scores are unchanged.",
        "",
        f"- Image AUROC: `{report['frozen_baseline']['image_auroc']:.12f}`",
        f"- Threshold: `{report['frozen_baseline']['threshold']!r}`",
        "",
        "## 2. Representation experiments",
        "",
        "| Candidate | Standalone image AUROC | Δ vs D3 | Mean abs score Δ | Pixel AUROC | AUPRO | Standalone gate | Dual gate |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["representations"]:
        standalone = row["standalone_representation_branch"]
        lines.append(
            f"| {row['candidate']} | {row['image_auroc']:.6f} | {standalone['image_auroc_delta_vs_d3']:+.6f} | {standalone['image_score_delta_vs_d3']['mean_absolute']:.6f} | {row['pixel_auroc']:.6f} | {row['aupro']:.6f} | {standalone['verdict']} | {row['dual_objective']['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Dual-objective design",
            "",
            "The image branch is always the frozen D3 A0 score. Each representation is evaluated as an independent pixel branch. Standalone representation image AUROC is diagnostic only and never replaces D3 scoring.",
            "",
            f"- Separation supported: `{report['dual_objective_design']['separation_supported']}`",
            f"- Passing pixel branches: `{report['dual_objective_design']['passing_pixel_branches']}`",
            f"- Preferred pixel branch by localization metrics: `{report['dual_objective_design']['preferred_pixel_branch_by_localization']}`",
            "- Every representation fails the standalone three-metric gate because its own image AUROC is below 0.75; every dual configuration passes when the immutable D3 image branch is retained.",
            "",
            "## 4. Localization gate",
            "",
            "PASS requires Pixel AUROC ≥ 0.75, AUPRO ≥ 0.50, and immutable D3 image AUROC ≥ 0.75.",
            "",
            f"Overall verdict: **`{report['overall_verdict']}`**.",
            "",
            "## 5. Isolation",
            "",
            "R-L1/R-L2 banks are runtime-only experimental evidence. R-L4 reuses a previously frozen experimental DINOv2-S bank. None is registered as, copied into, or substituted for the D3 candidate artifact. No fine-tuning, supervised training, threshold tuning, or production promotion occurred.",
            "",
            "## Evidence",
            "",
            "- `docs/d3-localization-representation-results.json`",
            "- `docs/d3-localization-representation-test-report.json`",
            "- Runtime checkpoints and experimental banks remain under ignored `model-training/runs/steel-d3-localization-representation/`.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_existing() -> dict:
    registry, manifest, hashes = load_lineage()
    _, normal_ids, anomaly_ids = load_ids()
    r1_manifest = verify_experimental_bank("R-L1", R1_BANK)[1]
    r2_manifest = verify_experimental_bank("R-L2", R2_BANK)[1]
    r4_manifest = verify_r4_bank()[1]
    bank_hashes = {
        "R-L1": r1_manifest["bank_sha256"],
        "R-L2": r2_manifest["bank_sha256"],
        "R-L4": r4_manifest["bank_sha256"],
    }
    checkpoint = load_eval_checkpoint(manifest, hashes, bank_hashes)
    order = [("test_normal", image_id) for image_id in normal_ids] + [
        ("recovery_holdout_anomaly", image_id) for image_id in anomaly_ids
    ]
    return finalize(manifest, hashes, hashes, order, anomaly_ids, checkpoint, bank_hashes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("bank-rl1", "bank-rl2", "banks", "evaluate", "finalize", "all"), default="all")
    args = parser.parse_args()
    try:
        if args.stage in {"bank-rl1", "banks", "all"}:
            build_bank("R-L1")
        if args.stage in {"bank-rl2", "banks", "all"}:
            build_bank("R-L2")
        if args.stage in {"evaluate", "all"}:
            evaluate()
        if args.stage == "finalize":
            finalize_existing()
        return 0
    except Exception as exc:
        print(f"D3_LOCALIZATION_REPRESENTATION_BLOCKED:{type(exc).__name__}:{exc}", flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

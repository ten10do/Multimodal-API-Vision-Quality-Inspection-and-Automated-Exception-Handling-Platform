"""Train steel-domain PatchCore memory bank (Q/R items).

Keeps Phase 6 baseline method: WideResNet-50-2, layer2+layer3 (1536-d),
unit-normalized features, deterministic seed, normal-only memory bank
(random subsample of 50k patches, equivalent to Phase 6 rng.choice semantics
via reservoir sampling to bound memory).

Deliberately NO: FAISS, PCA, ANN optimization, backbone change, speed tuning.

Memory bank source: train_normal ONLY. Threshold (R item):
  threshold = max(train_normal original-image scores)
  original-image score = max over the 7 tile scores.

New identity: model_name=steel-patchcore, model_version=1.0.0.
Does NOT touch inference-service/models/patchcore-bottle (mvtec baseline).

Usage:
  python inference-service/scripts/train_steel_patchcore.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_app.patchcore_predictor import PatchCorePredictor  # noqa: E402
from steel_patchcore.tile import TILE_X0, TILE, tile_coords  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "model-training/datasets/severstal-steel"
SPLIT = DS / "split_manifest.json"
IMG_DIR = DS / "raw/train_images"
OUT_DIR = ROOT / "inference-service/models/steel-patchcore"
CKPT = DS / "raw/steel_train_ckpt.npz"   # resume checkpoint (gitignored raw)
THR_CKPT = DS / "raw/steel_threshold_ckpt.json"
THR_CKPT_PREV = DS / "raw/steel_threshold_ckpt.prev.json"

BANK_PATCHES = 50_000
SEED = 42
IMAGE_SIZE = 256  # tile size; PatchCore input matches the tile, no resize
MODEL_NAME = "steel-patchcore"
MODEL_VERSION = "1.0.0"


def save_feature_ckpt(reservoir: np.ndarray, seen: int, done_ids: list[str]) -> None:
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CKPT, reservoir=reservoir, seen=seen, done_ids=np.asarray(done_ids, dtype=object))
    print(f"checkpoint saved: seen={seen} done={len(done_ids)}", flush=True)


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON via a unique temp file + flush/fsync + validate + atomic replace."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    # validate before committing
    json.load(open(tmp, encoding="utf-8"))
    os.replace(tmp, path)  # atomic on the same volume


def save_threshold_ckpt(done_ids: list[str], scores: list[float], current_max: float,
                        next_index: int, bank_sha: str, split_sha: str) -> None:
    """Durably persist threshold progress: current + previous copy.

    The previous valid checkpoint is only advanced AFTER a new checkpoint has
    been atomically written and validated.
    """
    import os

    # promote current -> prev only if current is a valid checkpoint
    if THR_CKPT.exists():
        try:
            json.load(open(THR_CKPT, encoding="utf-8"))
            os.replace(THR_CKPT, THR_CKPT_PREV)
        except Exception:  # noqa: BLE001
            pass
    data = {
        "done_ids": done_ids,
        "scores": scores,
        "current_max": current_max,
        "next_index": next_index,
        "bank_sha256": bank_sha,
        "split_sha256": split_sha,
    }
    atomic_write_json(THR_CKPT, data)


def load_threshold_ckpt() -> dict | None:
    """Resume from current; on corruption, fall back to the previous copy.

    Accepts both new (done_ids/scores) and legacy (image_ids/scores) formats.
    """
    for p in (THR_CKPT, THR_CKPT_PREV):
        if not p.exists():
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
            done_ids = d.get("done_ids") or d.get("image_ids")
            scores = d.get("scores")
            assert isinstance(done_ids, list) and isinstance(scores, list)
            assert len(done_ids) == len(scores)
            return {"done_ids": list(done_ids), "scores": [float(s) for s in scores],
                    "bank_sha256": d.get("bank_sha256"),
                    "split_sha256": d.get("split_sha256")}
        except Exception:  # noqa: BLE001
            continue
    return None


def bootstrap_threshold_ckpt(loaded: dict, bank_sha: str, split_sha: str) -> None:
    """Migrate a legacy-format checkpoint into the durable two-copy format.

    Validates completeness (equal-length done_ids/scores) and binds the
    checkpoint to the current bank/split hashes. The previous generation is
    preserved as prev BEFORE writing the new current, so current and prev never
    reference the same generation.
    """
    import os

    done_ids = loaded["done_ids"]
    scores = loaded["scores"]
    assert len(done_ids) == len(scores) and len(done_ids) > 0, "legacy checkpoint incomplete"
    # preserve previous generation before writing the new current
    if THR_CKPT.exists():
        try:
            json.load(open(THR_CKPT, encoding="utf-8"))
            os.replace(THR_CKPT, THR_CKPT_PREV)
        except Exception:  # noqa: BLE001
            pass
    data = {
        "done_ids": done_ids,
        "scores": scores,
        "current_max": max(scores) if scores else 0.0,
        "next_index": len(done_ids),
        "bank_sha256": bank_sha,
        "split_sha256": split_sha,
    }
    atomic_write_json(THR_CKPT, data)
    print(f"threshold checkpoint migrated: {len(done_ids)} done, "
          f"bank_sha={bank_sha[:8]}... split_sha={split_sha[:8]}...", flush=True)



def load_feature_ckpt() -> tuple[np.ndarray | None, int, list[str]]:
    if CKPT.exists():
        d = np.load(CKPT, allow_pickle=True)
        return d["reservoir"].astype(np.float32), int(d["seen"]), list(d["done_ids"])
    return None, 0, []


BANK_PATH = OUT_DIR / "bank.npz"
META_PATH = OUT_DIR / "bank_meta.json"


def sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_sha_at_seal() -> str:
    """The bank_sha256 recorded when the bank was SEALED (immutable reference)."""
    meta = json.load(open(META_PATH, encoding="utf-8"))
    return meta["bank_sha256"]


def split_sha_at_train() -> str:
    """SHA256 of split_manifest.json at training time (calibration provenance)."""
    return sha256_file(SPLIT)


def finalize_bank(reservoir: np.ndarray, train_ids: list[str], candidate_patches: int) -> tuple[np.ndarray, dict]:
    """Seal the memory bank artifact once feature extraction is complete.

    Feature extraction becomes SEALED after this; the threshold stage may only
    READ this artifact and never re-run extraction.
    """
    assert reservoir.shape == (BANK_PATCHES, 1536), f"shape {reservoir.shape}"
    assert bool(np.isfinite(reservoir).all()), "non-finite values in reservoir"
    sm = json.load(open(SPLIT, encoding="utf-8"))
    expected = sorted(sm["splits"]["train_normal"])
    assert sorted(train_ids) == expected, "bank source ids mismatch split train_normal"
    assert len(set(train_ids)) == len(train_ids) == len(expected), "duplicate or missing source ids"
    split_sha = sha256_file(SPLIT)
    config = {
        "bank_patches": BANK_PATCHES,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "candidate_patches": candidate_patches,
        "source_train_normal": len(train_ids),
        "sampling": "reservoir (uniform, no replacement semantics, seed 42)",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BANK_PATH,
        features=reservoir,
        threshold=0.0,  # placeholder; final threshold written after scoring
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        train_images=len(train_ids),
        bank_patches=BANK_PATCHES,
        seed=SEED,
        image_size=IMAGE_SIZE,
    )
    bank_sha = sha256_file(BANK_PATH)
    meta = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "bank_path": str(BANK_PATH),
        "bank_sha256": bank_sha,
        "feature_dim": 1536,
        "row_count": int(reservoir.shape[0]),
        "finite_only": True,
        "split_manifest_sha256": split_sha,
        "config": config,
    }
    json.dump(meta, open(META_PATH, "w"), indent=2)
    print("BANK_SEALED", json.dumps({k: v for k, v in meta.items() if k != "bank_path"}), flush=True)
    return reservoir, meta


def load_tiles(image_id: str) -> list[Image.Image]:
    img = Image.open(IMG_DIR / f"{image_id}.jpg")
    if img.mode != "RGB":
        img = img.convert("RGB")
    tiles = []
    for tid in range(len(TILE_X0)):
        x0, y0, w, h = tile_coords(tid)
        tiles.append(img.crop((x0, y0, x0 + w, y0 + h)))
    return tiles


def main() -> int:
    if not SPLIT.exists():
        print("TRAIN_BLOCKED: split_manifest.json missing")
        return 2
    sm = json.load(open(SPLIT, encoding="utf-8"))
    train_ids = sm["splits"]["train_normal"]
    print(f"train_normal images: {len(train_ids)}")

    # ---- Process Lifecycle Gate ----
    from steel_patchcore.lifecycle import lifecycle_enter
    _bank_sha = meta_sha_at_seal() if META_PATH.exists() else "pending"
    _rc = lifecycle_enter("train", _bank_sha, THR_CKPT, THR_CKPT.parent)
    if _rc is not None:
        return _rc

    predictor = PatchCorePredictor(image_size=IMAGE_SIZE)
    predictor._ensure_model()
    print("device:", predictor.device)

    # ---- feature extraction with reservoir sampling (bounded memory) ----
    reservoir, seen, done_ids = load_feature_ckpt()
    if reservoir is None:
        reservoir = np.zeros((BANK_PATCHES, 1536), dtype=np.float32)
        seen = 0
        done_ids = []
    done_set = set(done_ids)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    todo = [img_id for img_id in train_ids if img_id not in done_set]
    print(f"resume: done={len(done_ids)} todo={len(todo)} seen={seen}", flush=True)
    processed = len(done_ids)
    for i, img_id in enumerate(todo, 1):
        for tile in load_tiles(img_id):
            feats = predictor._embed(tile)  # [1024, 1536]
            for f in feats:
                seen += 1
                if seen <= BANK_PATCHES:
                    reservoir[seen - 1] = f
                else:
                    j = rng.integers(0, seen)
                    if j < BANK_PATCHES:
                        reservoir[j] = f
        done_ids.append(img_id)
        processed += 1
        if processed % 200 == 0:
            el = time.perf_counter() - t0
            print(f"...{processed}/{len(train_ids)} images, patches seen={seen}, {el:.0f}s", flush=True)
            save_feature_ckpt(reservoir, seen, done_ids)
    print(f"feature extraction done: {seen} patches -> reservoir {BANK_PATCHES}")
    save_feature_ckpt(reservoir, seen, done_ids)
    # SEAL the bank: after this point threshold may only READ bank.npz
    bank, _meta = finalize_bank(reservoir, train_ids, candidate_patches=seen)

    # ---- threshold on train_normal ORIGINAL-image scores (resumable) ----
    # load the sealed bank from disk (never re-run feature extraction)
    bank = np.load(BANK_PATH)["features"]
    bank_sha = meta_sha_at_seal()
    split_sha = split_sha_at_train()
    td = load_threshold_ckpt()
    image_scores: list[float] = []
    thr_done: list[str] = []
    if td is not None:
        image_scores = [float(s) for s in td["scores"]]
        thr_done = list(td["done_ids"])
        if not td.get("bank_sha256"):
            # legacy checkpoint: validate + bind hashes + migrate to durable format
            bootstrap_threshold_ckpt(td, bank_sha, split_sha)
        print(f"threshold resume from checkpoint (bank_sha={td.get('bank_sha256','?')[:8] if td.get('bank_sha256') else 'migrated'}...)",
              flush=True)
    predictor._bank = bank
    predictor._threshold = 0.0
    thr_todo = [img_id for img_id in train_ids if img_id not in set(thr_done)]
    print(f"threshold resume: done={len(thr_done)} todo={len(thr_todo)}", flush=True)
    for img_id in thr_todo:
        tile_scores = []
        for tile in load_tiles(img_id):
            _, s = predictor.score(tile)
            tile_scores.append(s)
        image_scores.append(float(max(tile_scores)))
        thr_done.append(img_id)
        if len(thr_done) % 200 == 0:
            save_threshold_ckpt(thr_done, image_scores, max(image_scores),
                                len(thr_done), bank_sha, split_sha)
            print(f"threshold ...{len(thr_done)}/{len(train_ids)}", flush=True)
    save_threshold_ckpt(thr_done, image_scores, max(image_scores),
                        len(thr_done), bank_sha, split_sha)
    # ---- final verification: train_normal must be fully and uniquely scored ----
    order = {img: s for img, s in zip(thr_done, image_scores)}
    image_scores = [order[img] for img in train_ids]
    expected_n = len(train_ids)
    scored_n = len(thr_done)
    unique_n = len(set(thr_done))
    missing = [i for i in train_ids if i not in order]
    duplicates = int(scored_n - unique_n)
    print(f"verify: expected={expected_n} scored={scored_n} unique={unique_n} "
          f"missing={len(missing)} duplicates={duplicates}", flush=True)
    assert expected_n == scored_n == unique_n == 4721, "train_normal scoring incomplete"
    assert not missing and duplicates == 0, "missing or duplicate scored ids"
    threshold = float(max(image_scores))
    print(f"train normal original-image scores: min={min(image_scores):.4f} "
          f"p95={np.percentile(image_scores, 95):.4f} max={max(image_scores):.4f}")
    print(f"threshold (train normal max) = {threshold:.4f}")

    # ---- save artifacts (bank.npz is SEALED and MUST NOT be rewritten) ----
    bank_path = OUT_DIR / "bank.npz"
    bank_sha = sha256_file(bank_path)
    assert bank_sha == meta_sha_at_seal(), f"sealed bank modified: {bank_sha}"
    (DS / "bank_source.json").write_text(json.dumps(sorted(train_ids)))
    (DS / "train_normal_scores.json").write_text(json.dumps({
        "calibration_evidence": True,
        "purpose": "threshold calibration on train_normal original images only",
        "image_ids": train_ids,
        "original_image_scores": [round(s, 6) for s in image_scores],
        "score_definition": "max over 7 tile scores",
        "bank_sha256": bank_sha,
    }, indent=1))
    (DS / "threshold.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "threshold": round(threshold, 6),
        "threshold_method": "max(train_normal original-image scores); original score = max over 7 tile scores",
        "bank_sha256": bank_sha,
        "source_split_manifest_sha256": split_sha_at_train(),
        "scored_originals": scored_n,
        "expected_originals": expected_n,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_source": "train_normal_scores.json",
        "missing": len(missing),
        "duplicates": duplicates,
    }, indent=2))
    # clean up resume checkpoints after a successful full run
    for ck in (CKPT, THR_CKPT):
        try:
            ck.unlink()
        except OSError:
            pass
    print("saved:", bank_path, "threshold.json", "train_normal_scores.json")
    print("TRAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

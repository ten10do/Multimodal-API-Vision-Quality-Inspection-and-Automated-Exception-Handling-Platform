"""Spatial scale & local context GPU experiment for Steel PatchCore (Stage S/P).

Stage S: S0 (reference, =R2), S1 (layer3 + 3x3 avg context), S2 (layer3 + 5x5).
Spatial Context Gate. If it fails, Stage P: P0 (=S0), P1 (layer2 + 3x3),
Patch Scale Gate vs R1-layer2.

Reuses the frozen representation diagnostic subset, reservoir sampler
(seed 42, budget 50000), frozen tiling, and A0 aggregation. The strongest
single-layer (R2) and raw-layer2 (R1) references are read back from the
representation-phase results (their bank manifests are verified). No holdout,
no baseline-bank write, no sampling/backbone/tiling change.

Run: .venv-steel/Scripts/python.exe inference-service/scripts/run_steel_spatial_context_experiment.py
     [--stage spatial|patch|all]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model-training"))
sys.path.insert(0, str(ROOT / "inference-service"))

from steel_patchcore.recovery import sha256_file  # noqa: E402
from steel_patchcore.spatial_context import (  # noqa: E402
    BANK_BUDGET,
    PATCH_SCALE_CANDIDATES,
    PATCH_SCALE_GATE,
    SPATIAL_CONTEXT_CANDIDATES,
    SPATIAL_CONTEXT_GATE,
    SPATIAL_SEED,
    context_embed,
    patch_scale_gate_passed,
    select_best_spatial_candidate,
    spatial_context_gate_passed,
)

DS = ROOT / "model-training/datasets/severstal-steel"
SUBSET_MANIFEST = DS / "representation_diagnostic_manifest.json"
SUBSET_MANIFEST_SHA = DS / "representation_diagnostic_manifest.sha256"
FRZ = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EXPECTED_FROZEN_BANK_SHA = "291bb2903b6e9a3bc60ca4ae35380bd7cf312bc661366d10f87a75d3f0b8ebda"

RUN_ROOT = ROOT / "model-training/runs/steel-spatial-context"
OUT_JSON = RUN_ROOT / "results.json"
OUT_MD = RUN_ROOT / "results.md"
REPR_RESULTS = ROOT / "model-training/runs/steel-representation/results.json"
REPR_R2_BANK_MANIFEST = ROOT / "model-training/runs/steel-representation/R2/bank_manifest.json"
REPR_R1_BANK_MANIFEST = ROOT / "model-training/runs/steel-representation/R1/bank_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict) -> None:
    def sanitize(v):
        if isinstance(v, dict):
            return {k: sanitize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [sanitize(val) for val in v]
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def load_repr_module():
    script = ROOT / "inference-service/scripts/run_steel_representation_experiment.py"
    spec = importlib.util.spec_from_file_location("_repr_experiment", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def views_for_tile(h2: torch.Tensor, h3_up: torch.Tensor, candidate_ids):
    """Per-candidate segment embeddings for one tile (context then L2)."""
    out = {}
    for cid in candidate_ids:
        if cid in ("S0", "P0"):
            out[cid] = context_embed(h3_up, None)
        elif cid == "S1":
            out[cid] = context_embed(h3_up, 3)
        elif cid == "S2":
            out[cid] = context_embed(h3_up, 5)
        elif cid == "P1":
            out[cid] = context_embed(h2, 3)
    return out


def build_banks(module, model, device, image_ids, candidate_ids):
    """One forward pass per tile; build one reservoir per candidate."""
    rng = np.random.default_rng(SPATIAL_SEED)
    reservoirs = {}
    seen = 0
    for image_id in image_ids:
        tiles = module.load_normalized_tiles(module.IMG_DIR / f"{image_id}.jpg", device)
        for tile in tiles:
            h2, h3_up = module.extract_tile_spatial(model, tile)
            views = views_for_tile(h2, h3_up, candidate_ids)
            arrays = {cid: v.cpu().numpy().astype(np.float32) for cid, v in views.items()}
            dims = {cid: arr.shape[1] for cid, arr in arrays.items()}
            if not reservoirs:
                reservoirs = {cid: np.zeros((BANK_BUDGET, dims[cid]), dtype=np.float32) for cid in arrays}
            for index in range(arrays[candidate_ids[0]].shape[0]):
                seen += 1
                if seen <= BANK_BUDGET:
                    for cid, arr in arrays.items():
                        reservoirs[cid][seen - 1] = arr[index]
                else:
                    j = int(rng.integers(0, seen))
                    if j < BANK_BUDGET:
                        for cid, arr in arrays.items():
                            reservoirs[cid][j] = arr[index]
    return reservoirs


def score_candidates(module, model, device, banks, image_ids, candidate_ids):
    scores = {cid: [] for cid in candidate_ids}
    banks_t = {cid: torch.from_numpy(banks[cid]).to(device) for cid in candidate_ids}
    for image_id in image_ids:
        tile_max = {cid: [] for cid in candidate_ids}
        for tile in module.load_normalized_tiles(module.IMG_DIR / f"{image_id}.jpg", device):
            h2, h3_up = module.extract_tile_spatial(model, tile)
            views = views_for_tile(h2, h3_up, candidate_ids)
            for cid, emb in views.items():
                sim = emb @ banks_t[cid].T
                dist = 1.0 - sim.max(dim=1).values
                tile_max[cid].append(float(dist.max()))
        for cid in candidate_ids:
            scores[cid].append(max(tile_max[cid]))
    return scores


def load_repr_reference():
    """S0/P0 = R2, P-gate reference = R1, from the representation results."""
    if not REPR_RESULTS.exists():
        raise RuntimeError("representation results missing; run representation phase first")
    r = json.loads(REPR_RESULTS.read_text(encoding="utf-8"))
    cand = r["candidates"]
    r2 = cand["R2"]
    r1 = cand["R1"]
    r1_q1 = next(q["normal_vs_quartile_auroc"] for q in r1["quartiles"] if q["quartile"] == 1)
    s0 = {
        "image_auroc": float(r2["image_auroc"]),
        "quartiles": r2["quartiles"],
    }
    return {
        "s0_image_auroc": s0["image_auroc"],
        "s0_quartiles": r2["quartiles"],
        "r1_image_auroc": float(r1["image_auroc"]),
        "r1_q1_auroc": float(r1_q1),
    }


def run_stage_spatial(module, model, device, subset, ref):
    print("=== Stage S: S1/S2 local context ===", flush=True)
    train_ids = subset["train_normal"]
    val_ids = subset["validation_normal"]
    dev_ids = subset["dev_anomaly"]
    cids = ["S1", "S2"]
    banks = build_banks(module, model, device, train_ids, cids)
    for cid in cids:
        out_dir = RUN_ROOT / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        bank_path = out_dir / "bank.npz"
        np.savez_compressed(bank_path, features=banks[cid].astype(np.float32))
        sha = sha256_file(bank_path)
        atomic_write_json(out_dir / "bank_manifest.json", {
            "candidate_id": cid,
            "context": 3 if cid == "S1" else 5,
            "dim": int(banks[cid].shape[1]),
            "rows": int(banks[cid].shape[0]),
            "sampling": "reservoir Algorithm R",
            "seed": SPATIAL_SEED,
            "budget": BANK_BUDGET,
            "source": "train_normal diagnostic subset",
            "source_count": len(train_ids),
            "bank_sha256": sha,
            "subset_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
            "created_at": utc_now(),
        })
        print(f"{cid}: bank dim={banks[cid].shape[1]} sha={sha[:12]}", flush=True)

    print("scoring S1/S2", flush=True)
    train_scores = score_candidates(module, model, device, banks, train_ids, cids)
    val_scores = score_candidates(module, model, device, banks, val_ids, cids)
    dev_scores = score_candidates(module, model, device, banks, dev_ids, cids)
    ratios, quartiles = module.load_area_ratios(dev_ids)

    results = {}
    for cid in cids:
        ev = module.evaluate(train_scores[cid], val_scores[cid], dev_scores[cid], quartiles)
        q1 = next(q["normal_vs_quartile_auroc"] for q in ev["quartiles"] if q["quartile"] == 1)
        q2 = next(q["normal_vs_quartile_auroc"] for q in ev["quartiles"] if q["quartile"] == 2)
        passed = spatial_context_gate_passed(ref["s0_image_auroc"], ev["image_auroc"])
        results[cid] = {
            "image_auroc": ev["image_auroc"],
            "q1_auroc": q1,
            "q2_auroc": q2,
            "gate_passed": passed,
            **ev,
        }
    best = select_best_spatial_candidate(results)
    verdict = "SPATIAL_CONTEXT_SIGNAL_FOUND" if best else "SPATIAL_CONTEXT_GATE_FAILED"
    return {
        "verdict": verdict,
        "best_candidate": best,
        "s0_image_auroc": ref["s0_image_auroc"],
        "s0_quartiles": ref["s0_quartiles"],
        "candidates": results,
    }


def run_stage_patch(module, model, device, subset, ref, spatial):
    print("=== Stage P: P1 patch-scale diagnostic ===", flush=True)
    train_ids = subset["train_normal"]
    val_ids = subset["validation_normal"]
    dev_ids = subset["dev_anomaly"]
    cids = ["P1"]
    banks = build_banks(module, model, device, train_ids, cids)
    out_dir = RUN_ROOT / "P1"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "bank.npz"
    np.savez_compressed(bank_path, features=banks["P1"].astype(np.float32))
    sha = sha256_file(bank_path)
    atomic_write_json(out_dir / "bank_manifest.json", {
        "candidate_id": "P1", "context": 3, "dim": int(banks["P1"].shape[1]),
        "rows": int(banks["P1"].shape[0]), "sampling": "reservoir Algorithm R",
        "seed": SPATIAL_SEED, "budget": BANK_BUDGET,
        "source": "train_normal diagnostic subset", "source_count": len(train_ids),
        "bank_sha256": sha,
        "subset_manifest_sha256": SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip(),
        "created_at": utc_now(),
    })
    print(f"P1: bank dim={banks['P1'].shape[1]} sha={sha[:12]}", flush=True)
    train_scores = score_candidates(module, model, device, banks, train_ids, cids)
    val_scores = score_candidates(module, model, device, banks, val_ids, cids)
    dev_scores = score_candidates(module, model, device, banks, dev_ids, cids)
    ratios, quartiles = module.load_area_ratios(dev_ids)
    ev = module.evaluate(train_scores["P1"], val_scores["P1"], dev_scores["P1"], quartiles)
    q1 = next(q["normal_vs_quartile_auroc"] for q in ev["quartiles"] if q["quartile"] == 1)
    passed = patch_scale_gate_passed(ref["r1_image_auroc"], ref["r1_q1_auroc"], ev["image_auroc"], q1)
    verdict = "PATCH_SCALE_SIGNAL_FOUND" if passed else "PATCH_SCALE_GATE_FAILED"
    return {
        "verdict": verdict,
        "r1_image_auroc": ref["r1_image_auroc"],
        "r1_q1_auroc": ref["r1_q1_auroc"],
        "p1": {"image_auroc": ev["image_auroc"], "q1_auroc": q1, "gate_passed": passed, **ev},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["spatial", "patch", "all"], default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SPATIAL_CONTEXT_EXPERIMENT_REQUIRES_GPU")
    device = torch.device("cuda:0")

    from steel_patchcore.lifecycle import lifecycle_enter

    blocked = lifecycle_enter("spatial_context_s_p", "experimental_banks",
                              RUN_ROOT / "checkpoint.json", RUN_ROOT)
    if blocked is not None:
        return int(blocked)

    if sha256_file(FRZ) != EXPECTED_FROZEN_BANK_SHA:
        raise RuntimeError("FROZEN_BANK_SHA_MISMATCH")
    if sha256_file(SUBSET_MANIFEST) != SUBSET_MANIFEST_SHA.read_text(encoding="ascii").strip():
        raise RuntimeError("SUBSET_MANIFEST_SHA_MISMATCH")

    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    subset = {
        "train_normal": list(manifest["train_normal_subset"]),
        "validation_normal": list(manifest["validation_normal_subset"]),
        "dev_anomaly": list(manifest["recovery_dev_anomaly_subset"]),
    }
    assert manifest["holdout_access_count"] == 0

    module = load_repr_module()
    ref = load_repr_reference()
    model = module.build_model(device)

    staged = {}
    if args.stage in ("spatial", "all"):
        staged["spatial"] = run_stage_spatial(module, model, device, subset, ref)

    if args.stage == "all" and staged.get("spatial", {}).get("verdict") == "SPATIAL_CONTEXT_GATE_FAILED":
        staged["patch"] = run_stage_patch(module, model, device, subset, ref, staged["spatial"])
    elif args.stage == "patch":
        staged["patch"] = run_stage_patch(module, model, device, subset, ref, None)

    spatial = staged.get("spatial", {})
    patch = staged.get("patch")
    if patch and patch["verdict"] == "PATCH_SCALE_SIGNAL_FOUND":
        final = "PATCH_SCALE_SIGNAL_FOUND"
    elif spatial.get("verdict") == "SPATIAL_CONTEXT_SIGNAL_FOUND":
        final = "SPATIAL_CONTEXT_SIGNAL_FOUND"
    elif patch and patch["verdict"] == "PATCH_SCALE_GATE_FAILED":
        final = "SPATIAL_REPRESENTATION_GATE_FAILED"
    elif spatial.get("verdict") == "SPATIAL_CONTEXT_GATE_FAILED":
        final = "SPATIAL_CONTEXT_GATE_FAILED"
    else:
        final = "INCOMPLETE"

    payload = {
        "schema_version": "steel_patchcore_spatial_context_results_v1",
        "protocol_version": "spatial_context_protocol_v1",
        "verdict": final,
        "spatial": spatial or None,
        "patch": patch,
        "gates": {"spatial_context": SPATIAL_CONTEXT_GATE, "patch_scale": PATCH_SCALE_GATE},
        "candidate_definitions": {
            "spatial": SPATIAL_CONTEXT_CANDIDATES,
            "patch": PATCH_SCALE_CANDIDATES,
        },
        "holdout_access_count": 0,
        "generated_at": utc_now(),
    }
    atomic_write_json(OUT_JSON, payload)
    render_md(payload, OUT_MD)
    print(json.dumps({"final_verdict": final, "spatial_verdict": spatial.get("verdict"),
                      "patch_verdict": patch.get("verdict") if patch else None}, default=str))
    print(final)
    return 0


def render_md(payload: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# Steel PatchCore — Spatial Scale & Local Context Results")
    A("")
    A(f"Final verdict: `{payload['verdict']}`")
    A(f"- Generated at: {payload['generated_at']}")
    A(f"- Holdout access count: {payload['holdout_access_count']}")
    A("")
    s = payload["spatial"] or {}
    if s:
        A("## Stage S (S0/S1/S2)")
        A("")
        A(f"- S0 (reference = R2) Image AUROC = {s['s0_image_auroc']:.4f}")
        A(f"- Stage S verdict: `{s['verdict']}`; best: {s.get('best_candidate')}")
        A("")
        A("| Candidate | AUROC | Q1 | Q2 | Q3 | Q4 | Δ vs S0 | Gate |")
        A("|---|---|---|---|---|---|---|---|")
        A(f"| S0 | {s['s0_image_auroc']:.4f} | - | - | - | - | - | ref |")
        for cid in ("S1", "S2"):
            c = s["candidates"][cid]
            q = c["quartiles"]
            A(f"| {cid} | {c['image_auroc']:.4f} | {q[0]['normal_vs_quartile_auroc']:.4f} | {q[1]['normal_vs_quartile_auroc']:.4f} | {q[2]['normal_vs_quartile_auroc']:.4f} | {q[3]['normal_vs_quartile_auroc']:.4f} | {c['image_auroc'] - s['s0_image_auroc']:+.4f} | {'PASS' if c['gate_passed'] else 'FAIL'} |")
        A("")
    p = payload.get("patch")
    if p:
        A("## Stage P (P0/P1)")
        A("")
        A(f"- R1 (raw layer2) AUROC = {p['r1_image_auroc']:.4f}, Q1 = {p['r1_q1_auroc']:.4f}")
        A(f"- P1 AUROC = {p['p1']['image_auroc']:.4f} (Δ vs R1 {p['p1']['image_auroc'] - p['r1_image_auroc']:+.4f}), Q1 = {p['p1']['q1_auroc']:.4f} (Δ {p['p1']['q1_auroc'] - p['r1_q1_auroc']:+.4f})")
        A(f"- Stage P verdict: `{p['verdict']}`")
        A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
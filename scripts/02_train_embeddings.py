"""KG 임베딩 학습 후 엔티티 임베딩 행렬 저장 (TransE / Ours).

    python scripts/02_train_embeddings.py                    # 2모델 × 3seed (약 5분)
    python scripts/02_train_embeddings.py --epochs 5 --seeds 42 --out /tmp/emb   # 스모크

결과 : embeddings/<model>_s<seed>.npy  (|E| × d, entity2id 순서)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kgekit.data import load_kg  # noqa: E402
from kgekit.evaluate import evaluate  # noqa: E402
from kgekit.train import config_for, train_chunk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["TransE", "Ours"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="embeddings")
    args = ap.parse_args()
    kg = load_kg(ROOT / "data" / "processed")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        for s in args.seeds:
            p = out / f"{m}_s{s}.npy"
            if p.exists():
                continue
            t0 = time.time()
            model, P, _, _ = train_chunk(kg, kg.train, config_for(m, seed=s, epochs=args.epochs))
            np.save(p, P["E"].astype(np.float32))
            mrr = evaluate(model, P, kg, kg.test)["overall"]["mrr"]
            print(f"{m:6s} s{s}  link-prediction MRR={mrr:.4f}  → {p.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

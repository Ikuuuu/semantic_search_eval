"""의미 검색 평가 — 텍스트 / KG 임베딩 / 하이브리드 비교.

방법
  Text      : 엔티티 명칭 문자 n-gram TF-IDF + LSA (src/text_encoders.py, 외부 모델 불필요)
  KG-<모델> : 학습된 KG 엔티티 임베딩 (구조 정보)
  Hybrid    : [α·Text ; (1-α)·KG] 단위벡터 결합.  KG 밖 질의(T2)는 텍스트 최근접 엔티티의 KG 벡터를 빌려 쓴다
점수      : 코사인 유사도.  후보 = 조문을 제외한 모든 엔티티(법령·외부법령·기관·클래스), 질의 자신 제외
지표      : MRR (첫 정답 기준), Hits@1, Hits@10, Recall@10 (다중 정답)

    python scripts/03_evaluate.py                 # embeddings/ 의 모든 seed 평균
    python scripts/03_evaluate.py --alpha 0.3     # → results/semantic_search_a0.3.md
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.text_encoders import CharNgramEncoder  # noqa: E402

ARTICLE_RE = re.compile(r"제\d+(?:의\d+)?조$")
TASKS = ["T1_variant", "T2_abbrev", "T3_parent", "T4_citation"]


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def rank_metrics(scores, gold_idx):
    order = np.argsort(-scores)
    pos = {int(i): r for r, i in enumerate(order, 1)}
    ranks = sorted(pos[g] for g in gold_idx)
    return {"mrr": 1.0 / ranks[0], "hits@1": float(ranks[0] <= 1), "hits@10": float(ranks[0] <= 10),
            "recall@10": sum(r <= 10 for r in ranks) / len(ranks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5, help="Hybrid 의 텍스트 비중")
    ap.add_argument("--emb-dir", default="embeddings")
    args = ap.parse_args()

    names = [l.split("\t")[0] for l in sorted(
        (ROOT / "data/processed/entity2id.tsv").read_text(encoding="utf-8").splitlines(),
        key=lambda l: int(l.split("\t")[1]))]
    idx = {n: i for i, n in enumerate(names)}
    items = json.loads((ROOT / "data/eval/semantic_eval.json").read_text(encoding="utf-8"))
    cand = np.array([not ARTICLE_RE.search(n) for n in names])

    # 텍스트 인코더 : KG 엔티티 + KG 밖 질의를 같은 어휘 공간에서 적합
    extra = sorted({it["query"] for it in items if it["query"] not in idx})
    enc = CharNgramEncoder(dim=100)
    Tall = enc.encode(names + extra)
    T, Tq = Tall[: len(names)], {q: Tall[len(names) + i] for i, q in enumerate(extra)}

    embs = defaultdict(list)
    for p in sorted((ROOT / args.emb_dir).glob("*_s*.npy")):
        embs[p.stem.split("_s")[0]].append(unit(np.load(p).astype(np.float64)))
    if not embs:
        print("※ embeddings/ 가 비어 있어 Text 방법만 평가합니다 (scripts/02_train_embeddings.py 실행 권장)")

    def evaluate(method, Ks):
        """Ks : seed 별 KG 행렬 리스트 (Text 는 [None])."""
        per_task = defaultdict(list)
        for K in Ks:
            if method == "Text":
                M = T
            elif method.startswith("KG-"):
                M = K
            else:
                M = unit(np.hstack([args.alpha * T, (1 - args.alpha) * K]))
            for it in items:
                gold = [idx[g] for g in it["gold"] if g in idx]
                if not gold:
                    continue
                q = it["query"]
                if q in idx:
                    qv = M[idx[q]]
                elif method == "Text":
                    qv = Tq[q]
                elif method.startswith("KG-"):
                    continue                       # KG 밖 질의는 KG 단독으로 표현 불가
                else:                              # Hybrid : 텍스트 최근접 엔티티의 KG 벡터 차용
                    anchor = int(np.argmax(np.where(cand, T @ Tq[q], -np.inf)))
                    qv = unit(np.hstack([args.alpha * Tq[q], (1 - args.alpha) * K[anchor]]))
                s = M @ qv
                s = np.where(cand, s, -np.inf)
                if q in idx:
                    s[idx[q]] = -np.inf
                per_task[it["task"]].append(rank_metrics(s, gold))
        return {t: {k: float(np.mean([m[k] for m in v])) for k in v[0]} | {"n": len(v) // len(Ks)}
                for t, v in per_task.items()}

    methods = {"Text": [None]}
    for m, Ks in embs.items():
        methods[f"KG-{m}"] = Ks
        methods[f"Hybrid(Text+{m})"] = Ks
    results = {name: evaluate(name, Ks) for name, Ks in methods.items()}
    (ROOT / "results").mkdir(exist_ok=True)
    tag = "" if abs(args.alpha - 0.5) < 1e-9 else f"_a{args.alpha:g}"
    (ROOT / "results" / f"semantic_search{tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    seeds = max((len(v) for v in embs.values()), default=0)
    L = [f"# 의미 검색 평가 결과 (α={args.alpha}, KG 임베딩 {seeds} seed 평균)", "",
         "| 방법 | " + " | ".join(f"{t} MRR / H@10" for t in TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for name, res in results.items():
        cells = [f"{res[t]['mrr']:.3f} / {res[t]['hits@10']:.3f}" if t in res else "–" for t in TASKS]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    n = {t: next((r[t]["n"] for r in results.values() if t in r), 0) for t in TASKS}
    L += ["", "질의 수 : " + ", ".join(f"{t}={n[t]}" for t in TASKS),
          "", "T4 는 정답이 여러 개이므로 Recall@10 도 함께 본다 :", ""]
    L += [f"- {name} : Recall@10 = {res['T4_citation']['recall@10']:.3f}"
          for name, res in results.items() if "T4_citation" in res]
    (ROOT / "results" / f"semantic_search{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""의미 검색 평가셋 생성 (국방 법령 KG v1.0 기준).

과제 4종 — 모두 "질의 → 법령/기관 엔티티 순위" 형태 (조문 노드는 후보에서 제외)
  T1 표기변이 : PDF 추출 공백 오류·가운뎃점 변이로 갈라진 노드 → 대표 노드     (KG 안 질의, 자동 생성)
  T2 약칭     : 사용자가 입력할 법한 약칭·통칭 → 정식 법령/기관               (KG 밖 텍스트 질의, 수작업 목록)
  T3 상위법령 : 하위 법령 → 상위(모법) 법령                                   (구조적 관련성, 자동 생성)
  T4 인용관계 : 말뭉치 법령 → 그 법령의 조문이 인용하는 다른 법령들 (다중 정답)   (구조적 관련성, 자동 생성)

    python scripts/01_build_evalset.py      → data/eval/semantic_eval.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DOTS = re.compile(r"[·ㆍ・•‧]")
ARTICLE_RE = re.compile(r"제\d+(?:의\d+)?조$")

# T2 : 국가법령정보센터 공식 약칭 + 실무 통칭 + 본 말뭉치 문서 내 정의어("이하 ○○이라 한다")
ABBREVIATIONS = [
    ("국가계약법", "국가를 당사자로 하는 계약에 관한 법률", "공식 약칭"),
    ("국가계약법 시행령", "국가를 당사자로 하는 계약에 관한 법률 시행령", "공식 약칭"),
    ("국가계약법 시행규칙", "국가를 당사자로 하는 계약에 관한 법률 시행규칙", "공식 약칭"),
    ("방산기술보호법", "방위산업기술 보호법", "공식 약칭"),
    ("방위산업발전법", "방위산업 발전 및 지원에 관한 법률", "공식 약칭"),
    ("정보공개법", "공공기관의 정보공개에 관한 법률", "공식 약칭"),
    ("중견기업법", "중견기업 성장촉진 및 경쟁력 강화에 관한 특별법", "공식 약칭"),
    ("판로지원법", "중소기업제품 구매촉진 및 판로지원에 관한 법률", "공식 약칭"),
    ("자본시장법", "자본시장과 금융투자업에 관한 법률", "공식 약칭"),
    ("신용정보법", "신용정보의 이용 및 보호에 관한 법률", "공식 약칭"),
    ("총포화약법", "총포ㆍ도검ㆍ화약류 등의 안전관리에 관한 법률", "공식 약칭"),
    ("외부감사법", "주식회사의 외부감사에 관한 법률", "공식 약칭"),
    ("국방과학기술혁신법", "국방과학기술혁신 촉진법", "공식 약칭(메타데이터)"),
    ("국방과학기술혁신법 시행령", "국방과학기술혁신 촉진법 시행령", "공식 약칭(메타데이터)"),
    ("방사청", "방위사업청", "실무 통칭"),
    ("촉진법", "국방과학기술혁신 촉진법", "문서 내 정의어"),
    ("방산원가규칙", "방산원가대상물자의 원가계산에 관한 규칙", "문서 내 정의어"),
    ("감독규칙", "방위사업감독관 직무 등에 관한 규칙", "문서 내 정의어"),
    ("과기출연기관법", "과학기술분야 정부출연연구기관 등의 설립ㆍ운영 및 육성에 관한법률", "문서 내 정의어"),
]


def key(s):
    return re.sub(r"\s+", "", _DOTS.sub("·", s))


def main():
    triples = [tuple(x) for x in json.loads((ROOT / "data" / "triples_all.json").read_text(encoding="utf-8"))]
    ents = {l.split("\t")[0] for l in (ROOT / "data/processed/entity2id.tsv").read_text(encoding="utf-8").splitlines()}
    etype = {h: t for h, r, t in triples if r == "type"}
    corpus = {h for h, t in etype.items() if t not in ("Article", "ExternalLegalDocument", "Organization")}
    deg = Counter()
    for h, _, t in triples:
        deg[h] += 1
        deg[t] += 1
    items = []

    # T1 표기변이
    groups = defaultdict(list)
    for e in sorted(ents):                       # 정렬 → 실행마다 같은 평가셋
        if not ARTICLE_RE.search(e) and e in etype:
            groups[key(e)].append(e)
    for _, g in sorted(groups.items()):
        if len(g) < 2:
            continue
        canon = next((e for e in g if e in corpus), None) or max(g, key=lambda e: (deg[e], e.count(" "), e))
        for q in g:
            if q != canon:
                items.append({"task": "T1_variant", "query": q, "gold": [canon], "in_kg": True})

    # T2 약칭
    for q, g, src in ABBREVIATIONS:
        if g in ents:
            items.append({"task": "T2_abbrev", "query": q, "gold": [g], "in_kg": q in ents, "source": src})

    # T3 상위법령
    parents = defaultdict(set)
    for h, r, t in triples:
        if r == "상위법령":
            parents[h].add(t)
    for c, ps in sorted(parents.items()):
        items.append({"task": "T3_parent", "query": c, "gold": sorted(ps), "in_kg": True})

    # T4 인용관계 (법령 단위로 집계, 자기 자신 제외)
    belongs = {h: t for h, r, t in triples if r == "소속"}
    cited = defaultdict(set)
    for h, r, t in triples:
        if r in ("인용", "외부인용") and h in belongs:
            src_law = belongs[h]
            tgt = belongs.get(t, t)          # 조문 → 조문 인용이면 대상 조문의 법령
            if tgt != src_law and not ARTICLE_RE.search(tgt):
                cited[src_law].add(tgt)
    for law, gs in sorted(cited.items()):
        if len(gs) >= 2:
            items.append({"task": "T4_citation", "query": law, "gold": sorted(gs), "in_kg": True})

    out = ROOT / "data" / "eval" / "semantic_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    c = Counter(it["task"] for it in items)
    print(f"→ {out.relative_to(ROOT)} : " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    missing = [q for q, g, _ in ABBREVIATIONS if g not in ents]
    if missing:
        print("  (T2 정답 엔티티가 KG 에 없어 제외:", missing, ")")


if __name__ == "__main__":
    main()

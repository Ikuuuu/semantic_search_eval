# 의미 검색 평가 결과 (α=0.5, KG 임베딩 3 seed 평균)

| 방법 | T1_variant MRR / H@10 | T2_abbrev MRR / H@10 | T3_parent MRR / H@10 | T4_citation MRR / H@10 |
|---|---|---|---|---|
| Text | 0.851 / 0.988 | 0.236 / 0.526 | 0.163 / 0.316 | 0.225 / 0.513 |
| KG-Ours | 0.042 / 0.070 | 0.007 / 0.000 | 0.175 / 0.368 | 0.304 / 0.573 |
| Hybrid(Text+Ours) | 0.544 / 0.826 | 0.148 / 0.421 | 0.273 / 0.474 | 0.330 / 0.692 |
| KG-TransE | 0.044 / 0.058 | 0.004 / 0.000 | 0.299 / 0.614 | 0.348 / 0.769 |
| Hybrid(Text+TransE) | 0.525 / 0.783 | 0.136 / 0.368 | 0.318 / 0.491 | 0.360 / 0.692 |

질의 수 : T1_variant=86, T2_abbrev=19, T3_parent=19, T4_citation=39

T4 는 정답이 여러 개이므로 Recall@10 도 함께 본다 :

- Text : Recall@10 = 0.088
- KG-Ours : Recall@10 = 0.092
- Hybrid(Text+Ours) : Recall@10 = 0.137
- KG-TransE : Recall@10 = 0.142
- Hybrid(Text+TransE) : Recall@10 = 0.141

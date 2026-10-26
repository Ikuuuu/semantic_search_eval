"""엔티티 명칭 → 고정 차원 벡터 인코더.

두 가지 인코더를 같은 인터페이스(`encode(names) -> (N, d) ndarray`)로 제공한다.

1. CharNgramEncoder (기본, 폐쇄망 대응)
   - 한국어 엔티티 명칭을 문자 n-gram(2~4) TF-IDF 로 표현한 뒤 TruncatedSVD(LSA)로 d 차원 축소.
   - 외부 모델 다운로드가 필요 없어 인터넷이 차단된 환경에서도 그대로 동작한다.
   - "방위사업법 시행령" 과 "방위사업법 제5조" 처럼 명칭을 공유하는 엔티티가 가까운 곳에서 출발한다.

2. HFEncoder (선택, KLUE-RoBERTa 등 사전학습 LM)
   - transformers 의 AutoModel 로 [CLS] / mean-pooling 표현을 얻고 PCA 로 d 차원 축소.
   - `pip install -r requirements-plm.txt` 및 HuggingFace 모델 접근이 필요하다.
"""
from __future__ import annotations

from typing import List

import numpy as np


def _unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


class CharNgramEncoder:
    name = "charngram"

    def __init__(self, dim: int = 100, ngram_range=(2, 4), seed: int = 0):
        self.dim = dim
        self.ngram_range = ngram_range
        self.seed = seed

    def encode(self, names: List[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                              sublinear_tf=True, min_df=1)
        X = vec.fit_transform(names)
        k = min(self.dim, X.shape[1] - 1)
        Z = TruncatedSVD(n_components=k, random_state=self.seed).fit_transform(X)
        if k < self.dim:  # 어휘가 매우 작을 때 0 패딩
            Z = np.hstack([Z, np.zeros((Z.shape[0], self.dim - k))])
        return _unit_rows(Z)


class HFEncoder:
    name = "klue"

    def __init__(self, dim: int = 100, model_name: str = "klue/roberta-base",
                 pooling: str = "mean", device: str = "cpu", batch_size: int = 64):
        self.dim = dim
        self.model_name = model_name
        self.pooling = pooling
        self.device = device
        self.batch_size = batch_size

    def encode(self, names: List[str]) -> np.ndarray:
        import torch
        from sklearn.decomposition import PCA
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(names), self.batch_size):
                enc = tok(names[i:i + self.batch_size], padding=True, truncation=True,
                          max_length=64, return_tensors="pt").to(self.device)
                hs = model(**enc).last_hidden_state
                if self.pooling == "cls":
                    v = hs[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    v = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
                outs.append(v.cpu().numpy())
        H = np.vstack(outs)
        Z = PCA(n_components=self.dim, random_state=0).fit_transform(H)
        return _unit_rows(Z)


def build_encoder(name: str, dim: int = 100, **kw):
    if name == "charngram":
        return CharNgramEncoder(dim=dim, **kw)
    if name == "klue":
        return HFEncoder(dim=dim, **kw)
    raise KeyError(f"unknown encoder: {name} (charngram | klue)")

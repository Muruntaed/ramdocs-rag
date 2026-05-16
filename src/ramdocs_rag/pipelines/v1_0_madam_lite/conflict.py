"""Cosine clustering of claims plus weighted majority vote.

Pure logic, no LLM calls. Hyperparameters match the legacy prototype
verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

from ramdocs_rag.core.retrieval import _get_embedder

from ...core.types import Claim

SIMILARITY_THRESHOLD: float = 0.75
MAJORITY_RATIO: float = 1.5


@dataclass(frozen=True)
class Cluster:
    members: tuple[str, ...]  # doc_ids
    representative_text: str
    weight: float


@dataclass(frozen=True)
class ConflictReport:
    clusters: tuple[Cluster, ...]
    has_conflict: bool
    winner: Cluster | None
    runner_up: Cluster | None
    ratio: float


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def cluster_claims(
    claims: list[Claim],
    reliability: dict[str, float],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[Cluster]:
    """Group ``supports`` claims whose pairwise cosine ≥ threshold. Pure function."""
    supports = [c for c in claims if c.stance == "supports" and c.text.strip()]
    if not supports:
        return []

    embedder = _get_embedder()
    texts = [_normalise(c.text) for c in supports]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    n = len(supports)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sim = embeddings @ embeddings.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: list[Cluster] = []
    for members in groups.values():
        ids = tuple(supports[i].doc_id for i in members)
        weight = float(sum(reliability.get(d, 0.0) for d in ids))
        rep_idx = max(members, key=lambda k: reliability.get(supports[k].doc_id, 0.0))
        clusters.append(
            Cluster(members=ids, representative_text=supports[rep_idx].text, weight=weight)
        )
    clusters.sort(key=lambda c: c.weight, reverse=True)
    return clusters


def detect_conflict(
    claims: list[Claim],
    reliability: dict[str, float],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    majority_ratio: float = MAJORITY_RATIO,
) -> ConflictReport:
    """Cluster the claims and check whether the top cluster beats the runner-up by the majority ratio."""
    clusters = cluster_claims(claims, reliability, similarity_threshold)
    if not clusters:
        return ConflictReport((), False, None, None, 0.0)
    if len(clusters) == 1:
        return ConflictReport(tuple(clusters), False, clusters[0], None, float("inf"))
    top, second = clusters[0], clusters[1]
    ratio = top.weight / second.weight if second.weight > 0 else float("inf")
    winner = top if ratio >= majority_ratio else None
    return ConflictReport(tuple(clusters), True, winner, second, ratio)


def minority_doc_ids(report: ConflictReport) -> set[str]:
    if report.winner is None:
        return set()
    winner_ids = set(report.winner.members)
    return {d for c in report.clusters for d in c.members if d not in winner_ids}

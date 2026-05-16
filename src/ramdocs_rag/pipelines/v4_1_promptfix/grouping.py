"""Deterministic entity grouping (introduced in v2.0, carried into v4).

Canonicalisation: lowercase, replace non-alphanumerics with spaces,
collapse whitespace. Parenthetical disambiguators are kept as part of
the key on purpose — they *are* the disambiguator. So
``"Placebo (album)"`` and ``"Placebo (band)"`` produce different keys,
while ``"Placebo (album)"`` and ``"placebo (album) "`` produce the same.

After canonicalisation we apply a soft substring merge: if ``key_A`` is
a substring of ``key_B`` the claims merge into the longer, more
specific group. That handles cases like ``"Without You I'm Nothing"``
vs ``"Without You I'm Nothing (Placebo album)"`` when one document
emits the short title and another emits the disambiguated form.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ramdocs_rag.core.types import Claim

_NON_ALNUM_RE = re.compile(r"[^a-z0-9()]+")


def canonicalize_entity(entity: str) -> str:
    """Canonicalise an entity string into a grouping key."""
    if not entity:
        return ""
    s = entity.lower().strip()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def group_by_entity(claims: list[Claim]) -> dict[str, list[Claim]]:
    """Group ``supports`` claims by their canonicalised entity.

    Applies a substring merge: if ``key_A`` is a substring of ``key_B``,
    claims from group ``A`` move into group ``B`` (the longer, more
    specific key wins).
    """
    raw_groups: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        if c.stance != "supports" or not c.entity:
            continue
        key = canonicalize_entity(c.entity)
        if key:
            raw_groups[key].append(c)

    keys = sorted(raw_groups.keys(), key=len, reverse=True)
    merged: dict[str, list[Claim]] = {}
    used: set[str] = set()
    for k in keys:
        if k in used:
            continue
        bucket = list(raw_groups[k])
        for other in keys:
            if other == k or other in used:
                continue
            if other in k and other != k:  # other is a strict substring of k
                # Anti-ambiguous-merge: if `other` is a substring of MORE than
                # one long key, it's an ambiguous identifier (e.g. "longtown"
                # appears in both "longtown scotland" AND "longtown oklahoma")
                # and we cannot merge it into any single group without risking
                # putting the wrong homonym into the wrong cluster. Leave it
                # as its own group instead. Saves multi-place / multi-homonym
                # docs whose analyzer was unable to recover a disambiguator.
                candidates = [
                    kk for kk in keys if kk != other and other in kk
                ]
                if len(candidates) > 1:
                    continue
                bucket.extend(raw_groups[other])
                used.add(other)
        merged[k] = bucket
        used.add(k)
    return merged


def display_entity(claims: list[Claim]) -> str:
    """Pick the most "complete" entity string for display (longest variant wins)."""
    return max((c.entity for c in claims if c.entity), key=len, default="")

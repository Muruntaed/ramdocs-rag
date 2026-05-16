# Work-in-progress

Active iteration tracker. This file is rewritten between sessions
(see rule 12 in `CLAUDE.md`) so that any agent / contributor picking up
the project can see where the last iteration stopped.

---

## (no active iteration)

Last completed milestone: **v4.1_promptfix · FROZEN (2026-05-16)** —
both freeze runs taken, every anchor metric met, baseline published in
`_baseline/v4.1_promptfix.json`, journal entry in `_journal.md`,
HTML page in `docs/journal/versions/v4.1_promptfix.html`.

**Open candidates for the next minor (v4.2) — not committed yet:**

1. **q306 Oklahoma group** (carried over from v4.1): the intra-mediator
   occasionally loses on a pure-table sub-line vs the main-line lead.
   Options:
   (a) give the analyzer an explicit `is_main_line_value=true/false`
   signal;
   (b) boost the lead-sentence weight inside
   `agents.resolve_entity_group`.
2. **`has_noise` F1 lift** (0.729 in v4.1 — the weakest category):
   the Skeptic occasionally fails to separate single-shape noise from
   a legitimate alternative answer. Handle with care — this is tied to
   `W_TRUST = 0.35`, and a naive reduction can bring back the v4.0
   misinfo regression.

**Next step — owner decision:** pick a direction (v4.2 / v5.0
parametric voice / something else) or close the project.

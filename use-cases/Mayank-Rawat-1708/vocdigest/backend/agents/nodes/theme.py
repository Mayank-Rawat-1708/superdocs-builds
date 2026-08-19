"""
@file: backend/agents/nodes/theme.py
@description: Groups conversations into themes. Clusters on the extracted issue phrases
    (which the LLM already normalised) rather than raw text, then asks the model to name
    and describe each cluster. Every theme carries evidence refs — file plus line for
    each supporting conversation — so no claim in the digest is unsourced.
@flow: run() -> load relevant conversations with facts -> greedy embedding clustering
    into candidate groups -> LLM names/merges the groups -> persist Theme rows with
    volume, centroid embedding, and evidence refs -> attach each conversation to its
    theme -> flag thin themes with a confidence note.
@dependencies:
    - backend.services.vector_store: centroid + cosine similarity for clustering
    - backend.services.groq_client.GroqClient: naming and merging clusters
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict

from sqlalchemy import select

from backend.agents.nodes.base import BaseNode, NodeSkip
from backend.agents.state import DigestState
from backend.config import settings
from backend.db.database import session_scope
from backend.models import Conversation, RunStatus, Theme
from backend.services.groq_client import GroqClient
from backend.services.heuristics import heuristic_name_clusters
from backend.services.llm_gate import run_with_fallback
from backend.services.vector_store import centroid, cosine_similarity

logger = logging.getLogger(__name__)

# Threshold lives in settings so it is tunable per deployment without a code change.
# The constructor argument still wins, which is how tests pin a value appropriate to
# the hash embedder.

# Themes below this many conversations get a confidence note rather than being asserted
# as a trend. Three is the smallest number where "pattern" is arguably honest.
MIN_CONFIDENT_VOLUME = 3

_SYSTEM = """\
You name clusters of customer-support conversations.

For each cluster you are given its member issue phrases. Return a short theme name and
a one-sentence description grounded in those phrases.

Return ONLY:
{"themes": [{"cluster": <int>, "name": "<3-6 words>", "description": "<one sentence>"}]}

Rules:
- The name must describe what the phrases actually say. Do not generalise beyond them.
- Do not invent a theme for a cluster you were not given.
- Prefer the customer's framing ("export silently fails") over internal jargon.
"""


def _greedy_cluster(
    items: list[tuple[uuid.UUID, list[float]]], threshold: float
) -> list[list[uuid.UUID]]:
    """Single-pass greedy clustering against running centroids.

    Chosen over k-means because the theme count is not known ahead of time and the
    result must be deterministic for idempotency — same input, same clusters, always.
    """
    clusters: list[list[uuid.UUID]] = []
    centroids: list[list[float]] = []
    members: list[list[list[float]]] = []

    for conv_id, vector in items:
        best_idx, best_sim = -1, -1.0
        for idx, cen in enumerate(centroids):
            sim = cosine_similarity(vector, cen)
            if sim > best_sim:
                best_idx, best_sim = idx, sim

        if best_idx >= 0 and best_sim >= threshold:
            clusters[best_idx].append(conv_id)
            members[best_idx].append(vector)
            new_centroid = centroid(members[best_idx])
            if new_centroid:
                centroids[best_idx] = new_centroid
        else:
            clusters.append([conv_id])
            centroids.append(list(vector))
            members.append([vector])

    return clusters


class ThemeNode(BaseNode):
    stage = "theme"
    running_status = RunStatus.THEMING

    def __init__(self, cluster_threshold: float | None = None) -> None:
        super().__init__()
        self.cluster_threshold = (
            cluster_threshold
            if cluster_threshold is not None
            else settings.cluster_threshold
        )

    async def run(self, state: DigestState, run_id: uuid.UUID) -> DigestState:
        async with session_scope() as session:
            rows = list(
                (
                    await session.execute(
                        select(Conversation).where(
                            Conversation.run_id == run_id,
                            Conversation.is_relevant.is_(True),
                            Conversation.embedding.is_not(None),
                        )
                    )
                ).scalars()
            )

            if not rows:
                raise NodeSkip(
                    "No embedded conversations available to cluster",
                    {"theme_count": 0, "theme_ids": []},
                )

            by_id = {c.id: c for c in rows}
            # Order by the input's natural key (file, line), NOT by row id. Row ids are
            # random UUIDs generated per run, so sorting by them made clustering stable
            # within a run but different between two runs over identical input — which
            # broke reproducibility. Greedy clustering is order-sensitive, so the
            # ordering must derive from the data, not from surrogate keys.
            items = [
                (c.id, list(c.embedding))
                for c in sorted(rows, key=lambda c: (c.source_file, c.source_line))
            ]
            clusters = _greedy_cluster(items, self.cluster_threshold)
            clusters.sort(key=len, reverse=True)

            # Ask the model to name each cluster from its members' issue phrases.
            cluster_phrases: list[list[str]] = []
            for members in clusters:
                phrases = []
                for cid in members[:12]:  # a sample is enough to name a cluster
                    facts = by_id[cid].extracted_facts or {}
                    phrase = facts.get("issue") or by_id[cid].raw_text[:120]
                    phrases.append(str(phrase))
                cluster_phrases.append(phrases)

            names: dict[int, dict[str, str]] = {}
            payload_in = "\n\n".join(
                f"Cluster {i} ({len(clusters[i])} conversations):\n"
                + "\n".join(f"- {p}" for p in phrases)
                for i, phrases in enumerate(cluster_phrases)
            )

            async def _llm():
                groq = GroqClient()
                try:
                    payload, usage, _ = await groq.complete_json(
                        _SYSTEM,
                        f"Name these {len(clusters)} clusters.",
                        untrusted_content=payload_in,
                        max_tokens=2048,
                    )
                    return payload, usage
                finally:
                    await groq.aclose()

            # Clustering itself is embedding-based and needs no model — only the naming
            # does. A Groq outage therefore costs label quality, not the analysis.
            gated = await run_with_fallback(
                "theme", _llm, lambda: heuristic_name_clusters(cluster_phrases)
            )
            payload = gated.data
            self.usage.merge(gated.usage)

            if isinstance(payload, dict):
                for item in payload.get("themes", []) or []:
                    try:
                        names[int(item["cluster"])] = {
                            "name": str(item.get("name", ""))[:200],
                            "description": str(item.get("description", ""))[:1000],
                        }
                    except (KeyError, TypeError, ValueError):
                        continue

            total_relevant = len(rows)
            theme_ids: list[str] = []
            thin_themes = 0

            for idx, members in enumerate(clusters):
                naming = names.get(idx) or {}
                # Fall back to the most common issue phrase if the model skipped this
                # cluster — an honest label beats an empty one.
                fallback = _most_common_phrase(cluster_phrases[idx])
                name = naming.get("name") or fallback
                description = naming.get("description") or (
                    f"Cluster of {len(members)} conversations about {fallback}."
                )

                vectors = [list(by_id[cid].embedding) for cid in members]
                evidence = [
                    {
                        "conversation_id": str(cid),
                        "file": by_id[cid].source_file,
                        "line": by_id[cid].source_line,
                        "citation": by_id[cid].citation,
                    }
                    for cid in members
                ]

                confidence_note = None
                if len(members) < MIN_CONFIDENT_VOLUME:
                    thin_themes += 1
                    confidence_note = (
                        f"Only {len(members)} conversation(s) support this theme. "
                        f"Too few to describe as a trend; reported for completeness."
                    )

                theme = Theme(
                    run_id=run_id,
                    name=name,
                    description=description,
                    volume_count=len(members),
                    volume_share=round(len(members) / total_relevant, 4),
                    embedding=centroid(vectors),
                    evidence_refs=evidence,
                    confidence_note=confidence_note,
                )
                session.add(theme)
                await session.flush()
                theme_ids.append(str(theme.id))

                for cid in members:
                    by_id[cid].theme_id = theme.id

            await session.flush()

        await self.log_decision(
            run_id,
            "CLUSTERED",
            f"Formed {len(clusters)} themes from {total_relevant} conversations "
            f"at similarity threshold {self.cluster_threshold}",
            {"themes": len(clusters), "thin_themes": thin_themes},
        )
        if thin_themes:
            await self.log_decision(
                run_id,
                "LOW_CONFIDENCE",
                f"{thin_themes} theme(s) have fewer than {MIN_CONFIDENT_VOLUME} "
                f"supporting conversations and are marked as non-trends",
                {"count": thin_themes},
            )

        if gated.degraded:
            await self.log_decision(
                run_id, "DEGRADED",
                f"Theme names were derived from word frequency, not a language model "
                f"({gated.reason}). Clustering itself is unaffected — it is "
                f"embedding-based and needs no model.",
                {"caveats": gated.caveats},
            )
            state["degraded_stages"] = sorted(
                set(state.get("degraded_stages") or []) | {"theme"}
            )
            state["degraded_caveats"] = list(
                dict.fromkeys((state.get("degraded_caveats") or []) + gated.caveats)
            )

        state["theme_count"] = len(clusters)
        state["theme_ids"] = theme_ids
        return state


def _most_common_phrase(phrases: list[str]) -> str:
    if not phrases:
        return "unlabelled issue"
    counts: dict[str, int] = defaultdict(int)
    for p in phrases:
        counts[p.strip().lower()] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0][:200]

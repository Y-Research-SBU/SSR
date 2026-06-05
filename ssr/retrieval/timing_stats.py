"""Retrieval timing statistics (index build vs query scoring)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalTimingStats:
    """Wall-clock breakdown for one ``retrieve`` / ``retrieve_from_bank`` call."""

    index_seconds: float = 0.0
    retrieve_seconds: float = 0.0
    n_queries: int = 0
    n_index_builds: int = 0
    score_seconds: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.score_seconds <= 0.0 and self.retrieve_seconds > 0.0:
            self.score_seconds = max(0.0, self.retrieve_seconds - self.index_seconds)

    @property
    def wall_seconds(self) -> float:
        """Index build + query scoring (full retrieve call)."""
        if self.retrieve_seconds > 0.0:
            return self.retrieve_seconds
        return self.index_seconds + self.score_seconds

    @property
    def retrieve_seconds_per_query(self) -> float:
        """Average phase-2 scoring time per query (excludes index build)."""
        if self.n_queries <= 0:
            return 0.0
        return self.score_seconds / self.n_queries

    @property
    def score_seconds_per_query(self) -> float:
        if self.n_queries <= 0:
            return 0.0
        return self.score_seconds / self.n_queries

    def as_dict(self) -> dict[str, float]:
        return {
            "retrieval_index_seconds": float(self.index_seconds),
            "retrieval_score_seconds": float(self.score_seconds),
            "retrieval_wall_seconds": float(self.wall_seconds),
            "retrieval_seconds_per_query": float(self.retrieve_seconds_per_query),
            "retrieval_score_seconds_per_query": float(self.score_seconds_per_query),
            "retrieval_n_queries": float(self.n_queries),
            "retrieval_n_index_builds": float(self.n_index_builds),
            # Legacy alias
            "retrieval_total_seconds": float(self.wall_seconds),
        }

    def log_summary(self, logger, *, prefix: str = "Retrieval timing") -> None:
        logger.info(
            "%s: index_build=%.3fs (%d shard-ingests) | retrieve_score=%.3fs | "
            "wall=%.3fs | per-query_score=%.4fs (n_queries=%d)",
            prefix,
            self.index_seconds,
            self.n_index_builds,
            self.score_seconds,
            self.wall_seconds,
            self.retrieve_seconds_per_query,
            self.n_queries,
        )

"""
graphite/claim_store.py — Persistent registry for Claims.

This provides an SQLite-backed store to persist extracted ATOMIC claims,
allowing them to be queried later by the Verification Engine.

Key semantics:
  - Claims are deduplicated by claim_id (deterministic hash of subject-predicate-object).
  - Evidence is accumulated: saving the same claim from a new source APPENDS evidence
    rather than overwriting it.
  - Dedupe policy:
      * Same source_id + same evidence_quote → skip (exact duplicate)
      * Same source_id, different quote → append
      * Different source_id → always append
"""

import json
import sqlite3
from typing import List, Optional, Set, Tuple

from .claim import Claim
from .schemas import Provenance


class ClaimStore:
    """A persistent SQLite store for Graphite Claims.

    Claims are deduplicated, evidence is accumulated. This makes the store
    a living fact base that grows stronger with each extraction run.
    """

    def __init__(self, db_path: str = "claims.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main claims table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                claim_text TEXT,
                claim_type TEXT,
                granularity TEXT,
                subject_entities TEXT,
                predicate TEXT,
                object_entities TEXT,
                as_of_date TEXT,
                computed_status TEXT,
                confidence_score REAL,
                full_json TEXT
            )
            """)

            # Index for fast querying
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(predicate)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_claims_as_of_date ON claims(as_of_date)"
            )

            conn.commit()

    # ── Evidence dedupe helpers ──

    @staticmethod
    def _evidence_key(ev: Provenance) -> Tuple[str, str]:
        """Fingerprint for evidence deduplication: (source_id, quote)."""
        return (ev.source_id, ev.evidence_quote.strip())

    @staticmethod
    def _merge_evidence(
        existing: List[Provenance],
        incoming: List[Provenance],
    ) -> List[Provenance]:
        """Merge incoming evidence into existing, skipping exact duplicates.

        Dedupe policy:
          - Same source_id + same evidence_quote → skip
          - Same source_id, different quote → append
          - Different source_id → always append
        """
        seen: Set[Tuple[str, str]] = {ClaimStore._evidence_key(ev) for ev in existing}
        merged = list(existing)
        for ev in incoming:
            key = ClaimStore._evidence_key(ev)
            if key not in seen:
                merged.append(ev)
                seen.add(key)
        return merged

    # ── Core CRUD ──

    def save_claim(self, claim: Claim) -> None:
        """Save a claim with evidence accumulation.

        If a claim with the same claim_id already exists, its evidence lists
        are merged (deduplicated) rather than overwritten. This makes
        repeated extraction runs additive instead of destructive.
        """
        existing = self.get_claim(claim.claim_id)

        if existing is not None:
            # Merge evidence from the new claim into the existing one
            existing.supporting_evidence = self._merge_evidence(
                existing.supporting_evidence,
                claim.supporting_evidence,
            )
            existing.weakening_evidence = self._merge_evidence(
                existing.weakening_evidence,
                claim.weakening_evidence,
            )
            # Update mutable fields from the newer claim
            existing.claim_text = claim.claim_text or existing.claim_text
            existing.as_of_date = claim.as_of_date or existing.as_of_date
            if claim.confidence is not None:
                existing.confidence = claim.confidence
            existing.computed_status = claim.computed_status
            # Preserve analyst override if the incoming claim has one
            if claim.is_overridden:
                existing.final_status = claim.final_status
                existing.override_reason = claim.override_reason
                existing.review_state = claim.review_state
                existing.reviewed_by = claim.reviewed_by
                existing.reviewed_at = claim.reviewed_at
                existing.reviewer_note = claim.reviewer_note
            # Preserve generator info if the new one provides it
            if claim.generator_id:
                existing.generator_id = claim.generator_id
            if claim.generation_metadata:
                existing.generation_metadata = {
                    **existing.generation_metadata,
                    **claim.generation_metadata,
                }
            claim = existing

        self._write_claim(claim)

    def _write_claim(self, claim: Claim) -> None:
        """Write a claim to SQLite (insert or replace)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            subjects_str = json.dumps(claim.subject_entities)
            objects_str = json.dumps(claim.object_entities)
            score = claim.confidence.score if claim.confidence else 0.0
            full_json = claim.model_dump_json()

            cursor.execute(
                """
            INSERT INTO claims (
                claim_id, claim_text, claim_type, granularity,
                subject_entities, predicate, object_entities,
                as_of_date, computed_status, confidence_score, full_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                claim_text=excluded.claim_text,
                as_of_date=excluded.as_of_date,
                computed_status=excluded.computed_status,
                confidence_score=excluded.confidence_score,
                full_json=excluded.full_json
            """,
                (
                    claim.claim_id,
                    claim.claim_text,
                    claim.claim_type.value,
                    claim.granularity.value,
                    subjects_str,
                    claim.predicate,
                    objects_str,
                    claim.as_of_date,
                    claim.computed_status.value,
                    score,
                    full_json,
                ),
            )

            conn.commit()

    def save_claims(self, claims: List[Claim]) -> None:
        """Batch save multiple claims with evidence accumulation."""
        for claim in claims:
            self.save_claim(claim)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by its exact ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT full_json FROM claims WHERE claim_id = ?", (claim_id,)
            )
            row = cursor.fetchone()

            if row:
                return Claim.model_validate_json(row[0])
            return None

    def search_claims(
        self,
        subject_contains: Optional[str] = None,
        object_contains: Optional[str] = None,
        predicate: Optional[str] = None,
        as_of_date: Optional[str] = None,
    ) -> List[Claim]:
        """Search claims by components."""
        query = "SELECT full_json FROM claims WHERE 1=1"
        params = []

        if subject_contains:
            query += " AND subject_entities LIKE ?"
            params.append(f"%{subject_contains}%")

        if object_contains:
            query += " AND object_entities LIKE ?"
            params.append(f"%{object_contains}%")

        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)

        if as_of_date:
            query += " AND as_of_date = ?"
            params.append(as_of_date)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [Claim.model_validate_json(row[0]) for row in rows]

    # ── Retrieval helpers ──

    def find_supporting_claims(self, claim: Claim) -> List[Claim]:
        """Find existing claims that support the given claim.

        A supporting claim shares at least one entity AND has the same
        predicate. Uses a single query instead of N+1 per entity.
        """
        conditions = []
        params = []

        for entity in claim.subject_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("subject_entities LIKE ?")
            params.append(f"%{key}%")

        for entity in claim.object_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("object_entities LIKE ?")
            params.append(f"%{key}%")

        if not conditions:
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT full_json FROM claims WHERE ({' OR '.join(conditions)})",
                params,
            )
            rows = cursor.fetchall()

        seen = set()
        supporting = []
        for row in rows:
            c = Claim.model_validate_json(row[0])
            if c.claim_id == claim.claim_id or c.claim_id in seen:
                continue
            seen.add(c.claim_id)
            if c.predicate == claim.predicate:
                supporting.append(c)

        return supporting

    def find_potential_conflicts(self, claim: Claim) -> List[Claim]:
        """Find claims that may conflict with the given claim.

        A potential conflict shares entities but has a different predicate.
        Uses a single query instead of N+1 per entity.
        """
        all_entities = claim.subject_entities + claim.object_entities
        conditions = []
        params = []

        # Search subjects in subject_entities column
        for entity in claim.subject_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("subject_entities LIKE ?")
            params.append(f"%{key}%")

        # Search objects in object_entities column
        for entity in claim.object_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("object_entities LIKE ?")
            params.append(f"%{key}%")

        # Also search reverse direction for conflicts
        for entity in claim.subject_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("object_entities LIKE ?")
            params.append(f"%{key}%")

        for entity in claim.object_entities:
            key = entity.split(":")[-1] if ":" in entity else entity
            conditions.append("subject_entities LIKE ?")
            params.append(f"%{key}%")

        if not conditions:
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT full_json FROM claims WHERE ({' OR '.join(conditions)})",
                params,
            )
            rows = cursor.fetchall()

        claim_entities = set(e.upper().strip() for e in all_entities)
        seen = set()
        conflicts = []
        for row in rows:
            c = Claim.model_validate_json(row[0])
            if c.claim_id == claim.claim_id or c.claim_id in seen:
                continue
            seen.add(c.claim_id)
            if c.predicate == claim.predicate:
                continue
            c_entities = set(
                e.upper().strip() for e in c.subject_entities + c.object_entities
            )
            if claim_entities & c_entities:
                conflicts.append(c)

        return conflicts

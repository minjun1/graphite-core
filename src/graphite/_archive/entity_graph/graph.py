"""
graphite/graph.py — Domain-agnostic graph storage layer.

Manages an in-memory NetworkX DiGraph with optional Neo4j and file-based loading.
Domain plugins configure what Cypher queries to run and how to
map Neo4j records to NetworkX edges.
"""

import hashlib
import html
import json
import math
import os
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import networkx as nx


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, x))


def generate_quote_hash(quote: str) -> str:
    """Generate a deterministic hash of a quote for evidence traceability."""
    if not quote:
        return ""
    cleaned = html.unescape(quote)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = " ".join(cleaned.split()).lower()
    return hashlib.sha256(cleaned.encode()).hexdigest()[:16]


class GraphStore:
    """
    Core graph storage engine.

    Supports two loading modes:
    1. File-based: load_from_file(path) — JSON or GraphML, no Neo4j needed
    2. Neo4j: load(queries, mappers) — full database integration

    Domain plugins provide:
    - node_query: Cypher query to load nodes
    - edge_query: Cypher query to load edges
    - node_mapper: Function to map Neo4j records to node attributes
    - edge_mapper: Function to map Neo4j records to edge attributes
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        # Lazy Neo4j — only connect when actually needed
        self._neo4j_uri = uri
        self._neo4j_user = user
        self._neo4j_password = password
        self._driver = None

        self.G = nx.DiGraph()
        self.loaded_at = 0
        self.graph_built_at = ""
        self.data_as_of = ""

        # Alias resolution (domain plugins can populate this)
        self._aliases: Dict[str, str] = {}
        self._node_meta: Dict[str, Dict] = {}

    @property
    def driver(self):
        """Lazy Neo4j driver — only connects when first accessed."""
        if self._driver is None:
            from neo4j import GraphDatabase

            uri = self._neo4j_uri or os.environ.get(
                "NEO4J_URI", "bolt://localhost:7687"
            )
            user = self._neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
            password = self._neo4j_password or os.environ.get("NEO4J_PASSWORD", "")
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        return self._driver

    def load_from_file(self, path: str) -> None:
        """Load graph from JSON or GraphML file. No Neo4j needed.

        Args:
            path: Path to a .json or .graphml graph file
        """
        from .io import load_graph

        self.G = load_graph(path)
        self.loaded_at = int(time.time())
        self.graph_built_at = self.G.graph.get(
            "built_at",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def load_aliases(self, aliases: Dict[str, str], meta: Dict[str, Dict] = None):
        """Load entity alias mapping (e.g., GOOG → GOOGL)."""
        self._aliases = {
            k.upper().strip(): v.upper().strip() for k, v in aliases.items()
        }
        if meta:
            self._node_meta = {k.upper().strip(): v for k, v in meta.items()}

    def canonicalize(self, identifier: str) -> str:
        """Resolve an identifier to its canonical form."""
        t = identifier.upper().strip()
        return self._aliases.get(t, t)

    def get_node_meta(self, identifier: str) -> dict:
        """Get metadata for a canonical node."""
        t = self.canonicalize(identifier)
        return self._node_meta.get(t, {})

    def close(self):
        if self._driver is not None:
            self._driver.close()

    def load(
        self,
        node_query: str,
        edge_query: str,
        node_mapper: Callable,
        edge_mapper: Callable,
        pre_load_hook: Callable = None,
    ):
        """
        Load graph from Neo4j using domain-provided queries and mappers.

        Args:
            node_query: Cypher query returning node records
            edge_query: Cypher query returning edge records
            node_mapper: fn(record) -> (node_id, attrs_dict) or None
            edge_mapper: fn(record, canonicalize_fn) -> (src, tgt, attrs_dict) or None
            pre_load_hook: Optional fn(session) called before loading nodes/edges
        """
        self.G.clear()

        with self.driver.session() as session:
            # Optional pre-load (e.g., build CIK→ticker map)
            if pre_load_hook:
                pre_load_hook(session)

            # Load nodes
            for record in session.run(node_query):
                result = node_mapper(record, self.canonicalize)
                if result is None:
                    continue
                node_id, attrs = result
                if node_id and not self.G.has_node(node_id):
                    self.G.add_node(node_id, **attrs)

            # Load edges
            for record in session.run(edge_query):
                result = edge_mapper(record, self.canonicalize)
                if result is None:
                    continue
                src, tgt, attrs = result
                if not src or not tgt or src == tgt:
                    continue

                # Keep edge with highest weight if duplicate
                if self.G.has_edge(src, tgt):
                    existing_w = self.G[src][tgt].get("bucket_weight", 0)
                    if attrs.get("bucket_weight", 0) > existing_w:
                        self.G[src][tgt].update(**attrs)
                else:
                    self.G.add_edge(src, tgt, **attrs)

        self.loaded_at = int(time.time())
        self.graph_built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def nodes(self):
        """Return sorted list of node dicts."""
        res = []
        for n, data in self.G.nodes(data=True):
            if n:
                res.append({"id": n, **data})
        return sorted(res, key=lambda x: x["id"])

    def stats(self):
        """Return graph statistics."""
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.G.number_of_edges(),
            "loaded_at": self.loaded_at,
            "graph_built_at": self.graph_built_at,
            "data_as_of": self.data_as_of,
        }

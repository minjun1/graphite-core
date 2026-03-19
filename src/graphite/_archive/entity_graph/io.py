"""
graphite/io.py — Save/load NetworkX graphs.

Serialization strategy:
  - GraphML: provenance stored as JSON string in edge attributes (interchange/debug)
  - JSON: full nested structure (rich storage)
  - Neo4j: provenance as edge property JSON (optional, requires neo4j extra)
"""
import json
import os
from typing import Optional

import networkx as nx


def save_graph(G: nx.DiGraph, path: str, format: str = "graphml") -> str:
    """Save a NetworkX graph to file.

    Args:
        G: NetworkX DiGraph to save
        path: Output file path
        format: "graphml" or "json"

    Returns:
        Actual path written to
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if format == "graphml":
        if not path.endswith(".graphml"):
            path += ".graphml"
        _sanitize_for_graphml(G)
        nx.write_graphml(G, path)
    elif format == "json":
        if not path.endswith(".json"):
            path += ".json"
        data = _graph_to_json(G)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return path


def load_graph(path: str) -> nx.DiGraph:
    """Load a NetworkX graph from file."""
    if path.endswith(".graphml"):
        G = nx.read_graphml(path)
        _deserialize_graphml_provenance(G)
        return G
    elif path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return _json_to_graph(data)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def _sanitize_for_graphml(G: nx.DiGraph) -> None:
    """Ensure all graph/node/edge attributes are GraphML-safe primitives."""
    for key in list(G.graph.keys()):
        if isinstance(G.graph[key], (dict, list)):
            G.graph[key] = json.dumps(G.graph[key], default=str)

    for _, data in G.nodes(data=True):
        for key in list(data.keys()):
            if isinstance(data[key], (dict, list)):
                data[key] = json.dumps(data[key], default=str)

    for _, _, data in G.edges(data=True):
        for key in list(data.keys()):
            if isinstance(data[key], (dict, list)):
                data[key] = json.dumps(data[key], default=str)


def _deserialize_graphml_provenance(G: nx.DiGraph) -> None:
    """Attempt to deserialize JSON strings back for provenance."""
    for _, _, data in G.edges(data=True):
        pj = data.get("provenance_json")
        if isinstance(pj, str):
            try:
                data["provenance_parsed"] = json.loads(pj)
            except (json.JSONDecodeError, TypeError):
                pass

    for key in ("edge_types", "assertion_modes"):
        val = G.graph.get(key)
        if isinstance(val, str):
            try:
                G.graph[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass


def _graph_to_json(G: nx.DiGraph) -> dict:
    """Convert graph to JSON-serializable dict with full nesting."""
    nodes = []
    for nid, data in G.nodes(data=True):
        node = {"id": nid}
        node.update(data)
        nodes.append(node)

    edges = []
    for src, tgt, data in G.edges(data=True):
        edge = {"source": src, "target": tgt}
        pj = data.get("provenance_json")
        edge_data = dict(data)
        if isinstance(pj, str):
            try:
                edge_data["provenance"] = json.loads(pj)
                del edge_data["provenance_json"]
            except (json.JSONDecodeError, TypeError):
                pass
        edge.update(edge_data)
        edges.append(edge)

    return {
        "metadata": dict(G.graph),
        "nodes": nodes,
        "edges": edges,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }


def _json_to_graph(data: dict) -> nx.DiGraph:
    """Convert JSON dict back to NetworkX graph."""
    G = nx.DiGraph()

    for k, v in data.get("metadata", {}).items():
        G.graph[k] = v

    for node in data.get("nodes", []):
        nid = node.pop("id")
        G.add_node(nid, **node)

    for edge in data.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        prov = edge.pop("provenance", None)
        if prov and isinstance(prov, list):
            edge["provenance_json"] = json.dumps(prov, default=str)
            edge["provenance_count"] = len(prov)
        G.add_edge(src, tgt, **edge)

    return G


def push_to_neo4j(
    G: nx.DiGraph,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> int:
    """Push graph to Neo4j. Provenance stored as JSON edge property.

    Returns:
        Number of edges written
    """
    from neo4j import GraphDatabase

    uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD", "")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    count = 0

    with driver.session() as session:
        for nid, data in G.nodes(data=True):
            node_type = data.get("node_type", "Entity")
            label = node_type.capitalize() if node_type else "Entity"
            props = {k: v for k, v in data.items()
                     if k != "node_type" and isinstance(v, (str, int, float, bool))}
            props["id"] = nid
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=nid, props=props,
            )

        for src, tgt, data in G.edges(data=True):
            edge_type = data.get("edge_type", "RELATES_TO")
            props = {k: v for k, v in data.items()
                     if k != "edge_type" and isinstance(v, (str, int, float, bool))}
            session.run(
                f"MATCH (a {{id: $src}}), (b {{id: $tgt}}) "
                f"MERGE (a)-[r:{edge_type}]->(b) SET r += $props",
                src=src, tgt=tgt, props=props,
            )
            count += 1

    driver.close()
    return count

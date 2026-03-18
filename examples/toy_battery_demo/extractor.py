"""
Toy extractor — deterministic keyword/regex extraction.

No LLM! This is purely regex-based to demonstrate Graphite's
pipeline structure without any external dependencies.
"""
import os
import re
from graphite import (
    ExtractedEdge, NodeRef, Provenance,
    SourceType, EdgeType, ConfidenceLevel, AssertionMode,
)


# ── Simple regex patterns ──

# "X accounting for approximately Y% of world output"
PRODUCES_PATTERN = re.compile(
    r"(?:the\s+)?([\w\s]+?)\s+(?:account\w*\s+for|dominat\w*|controll\w*|held)\s+"
    r"(?:approximately\s+)?(\d{1,3})%\s+of\s+(?:world|global)\s+"
    r"(?:[\w\s]*?)(?:production|output|refining|capacity)",
    re.IGNORECASE,
)

# "X sources Y from Z" or "Z is/are X's supplier"
SUPPLIES_PATTERN = re.compile(
    r"([\w\s]+?)\s+(?:sources?|procures?|purchases?|imports?)\s+"
    r"([\w\s]+?)\s+(?:from|through|via)\s+([\w\s]+?)(?:\s+under|\s+through|[.,])",
    re.IGNORECASE,
)

# "X supplies Y to Z" or "X is a supplier to Z"
SUPPLIES_TO_PATTERN = re.compile(
    r"([\w\s]+?)\s+(?:supplies?|delivers?|provides?|sells?)\s+"
    r"(?:[\w\s]+?)\s+(?:to|for)\s+([\w\s]+?)(?:\s+under|\s+for|[.,])",
    re.IGNORECASE,
)


# ── Country ISO mapping ──
COUNTRY_ISO = {
    "democratic republic of the congo": "CD", "drc": "CD", "congo": "CD",
    "australia": "AU", "philippines": "PH", "indonesia": "ID",
    "china": "CN", "chile": "CL", "finland": "FI", "belgium": "BE",
}

# ── Company ticker mapping ──
COMPANY_TICKER = {
    "catl": "CATL", "tesla": "TSLA", "glencore": "GLEN",
    "albemarle": "ALB", "albemarle corporation": "ALB",
    "sqm": "SQM", "bmw": "BMW", "volkswagen": "VOW",
    "lg energy solution": "LGES", "lg energy": "LGES",
    "bhp": "BHP", "contemporary amperex": "CATL",
}


def _normalize_entity(name: str):
    """Normalize an entity name → (NodeRef, is_country)."""
    clean = name.strip().lower()
    # Check if it's a country
    for country_name, iso in COUNTRY_ISO.items():
        if country_name in clean:
            return NodeRef.country(iso, label=name.strip()), True
    # Check if it's a known company
    for company_name, ticker in COMPANY_TICKER.items():
        if company_name in clean:
            return NodeRef.company(ticker, label=name.strip()), False
    # Unknown entity → treat as company
    short = re.sub(r'\s+', '_', name.strip().upper()[:20])
    return NodeRef.company(short, label=name.strip()), False


def extract_from_document(filepath: str) -> list[ExtractedEdge]:
    """Extract edges from a single text document using regex patterns."""
    with open(filepath) as f:
        text = f.read()

    filename = os.path.basename(filepath)
    edges = []

    # 1. PRODUCES: Country → Mineral production percentages
    for m in PRODUCES_PATTERN.finditer(text):
        entity_name, pct_str = m.group(1), m.group(2)
        entity_ref, is_country = _normalize_entity(entity_name)
        if not is_country:
            continue  # Only countries PRODUCE minerals

        # Detect which mineral from filename/context
        mineral = "COBALT"  # simplified: detect from filename
        if "lithium" in filepath.lower():
            mineral = "LITHIUM"

        edges.append(ExtractedEdge(
            from_node=entity_ref,
            to_node=NodeRef.mineral(mineral),
            edge_type=EdgeType.PRODUCES,
            assertion_mode=AssertionMode.EXTRACTED,
            attributes={"production_pct": float(pct_str), "bucket_weight": 0.7},
            provenance=[Provenance(
                source_id=filename,
                source_type=SourceType.PDF,
                evidence_quote=m.group(0).strip()[:280],
                confidence=ConfidenceLevel.HIGH,
            )],
        ))

    # 2. SUPPLIES_TO: Company → Company supply relationships
    for m in SUPPLIES_TO_PATTERN.finditer(text):
        supplier_name, buyer_name = m.group(1), m.group(2)
        supplier, _ = _normalize_entity(supplier_name)
        buyer, _ = _normalize_entity(buyer_name)
        if supplier.node_id == buyer.node_id:
            continue

        edges.append(ExtractedEdge(
            from_node=supplier,
            to_node=buyer,
            edge_type=EdgeType.SUPPLIES_TO,
            assertion_mode=AssertionMode.EXTRACTED,
            attributes={"bucket_weight": 0.5},
            provenance=[Provenance(
                source_id=filename,
                source_type=SourceType.PDF,
                evidence_quote=m.group(0).strip()[:280],
                confidence=ConfidenceLevel.MEDIUM,
            )],
        ))

    # 3. Sourcing patterns: X sources Y from Z
    for m in SUPPLIES_PATTERN.finditer(text):
        buyer_name, _material, supplier_name = m.group(1), m.group(2), m.group(3)
        supplier, _ = _normalize_entity(supplier_name)
        buyer, _ = _normalize_entity(buyer_name)
        if supplier.node_id == buyer.node_id:
            continue

        edges.append(ExtractedEdge(
            from_node=supplier,
            to_node=buyer,
            edge_type=EdgeType.SUPPLIES_TO,
            assertion_mode=AssertionMode.EXTRACTED,
            attributes={"bucket_weight": 0.5},
            provenance=[Provenance(
                source_id=filename,
                source_type=SourceType.PDF,
                evidence_quote=m.group(0).strip()[:280],
                confidence=ConfidenceLevel.MEDIUM,
            )],
        ))

    return edges


def extract_from_documents(doc_dir: str) -> list[ExtractedEdge]:
    """Extract edges from all .txt files in a directory."""
    all_edges = []
    for fname in sorted(os.listdir(doc_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(doc_dir, fname)
            edges = extract_from_document(path)
            print(f"  📄 {fname}: {len(edges)} edges")
            all_edges.extend(edges)
    return all_edges

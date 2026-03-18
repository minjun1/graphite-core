"""
graphite/text.py — Text processing with pluggable context strategies.

The extractor calls build_context(doc, strategy="...") to get
a keyword-scored, trimmed context string. New domains register
their own strategies.
"""
import hashlib
import re
from typing import Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════
# Strategy Registry
# ═══════════════════════════════════════

_strategies: Dict[str, Callable] = {}


def register_strategy(name: str, fn: Callable) -> None:
    """Register a context-building strategy.

    Args:
        name: Strategy name (e.g., "usgs_country_mineral", "sec_minerals")
        fn: Function(paragraphs, **kwargs) -> str
    """
    _strategies[name] = fn


def build_context(doc, strategy: str = "default", **kwargs) -> str:
    """Build LLM context using the named strategy.

    Called by extractors, NOT fetchers. The extractor chooses which
    strategy best suits its extraction task.

    Args:
        doc: DocumentContext with text_content and paragraphs
        strategy: Registered strategy name
        **kwargs: Passed through to strategy function

    Returns:
        Trimmed context string for LLM input
    """
    fn = _strategies.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown context strategy: '{strategy}'. Registered: {list(_strategies.keys())}")
    return fn(doc.paragraphs, **kwargs)


# ═══════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════

def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def normalize_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def clip_quote(s: str, max_chars: int = 280) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"

def split_into_paragraphs(
    text: str,
    min_len: int = 80,
    max_paras: int = 9000,
) -> List[str]:
    raw = text.split("\n\n")
    paras = []
    for p in raw:
        p = p.strip()
        if len(p) >= min_len:
            paras.append(p)
            if len(paras) >= max_paras:
                break
    return paras

def score_paragraph(paragraph: str, keywords: List[str]) -> int:
    lower = paragraph.lower()
    return sum(1 for kw in keywords if kw.lower() in lower)

def find_best_paragraph_for_quote(
    paragraphs: List[str],
    quote: str,
) -> Tuple[int, str]:
    """Find the paragraph that best matches a quote.

    Returns:
        (paragraph_index, paragraph_hash)
    """
    if not quote or not paragraphs:
        return -1, ""

    quote_lower = quote.lower().strip()
    best_idx = -1
    best_overlap = 0

    for i, p in enumerate(paragraphs):
        p_lower = p.lower()
        # Exact substring match
        if quote_lower in p_lower:
            return i, sha1_hex(p)[:12]

        # Word overlap score
        quote_words = set(quote_lower.split())
        para_words = set(p_lower.split())
        overlap = len(quote_words & para_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    if best_idx >= 0:
        return best_idx, sha1_hex(paragraphs[best_idx])[:12]
    return -1, ""


# ═══════════════════════════════════════
# Built-in Strategies
# ═══════════════════════════════════════

def _build_from_keywords(
    paragraphs: List[str],
    keywords: List[str],
    max_chars: int = 120_000,
    min_chars: int = 50_000,
    window: int = 2,
    max_anchors: int = 40,
) -> str:
    """Core strategy: keyword-scored paragraph selection with window expansion."""
    if not paragraphs:
        return ""

    scored = [(i, score_paragraph(p, keywords)) for i, p in enumerate(paragraphs)]
    scored.sort(key=lambda x: -x[1])

    anchors = [i for i, s in scored[:max_anchors] if s > 0]

    # Expand with surrounding context
    selected = set()
    for idx in anchors:
        for w in range(max(0, idx - window), min(len(paragraphs), idx + window + 1)):
            selected.add(w)

    # If not enough content, add more
    if not selected:
        selected = set(range(min(len(paragraphs), 50)))

    ordered = sorted(selected)
    parts = []
    total_chars = 0
    for i in ordered:
        p = paragraphs[i]
        tag = f"[PARA_{i}] {p}"
        if total_chars + len(tag) > max_chars:
            break
        parts.append(tag)
        total_chars += len(tag)

    # Pad if under min
    if total_chars < min_chars:
        for i, p in enumerate(paragraphs):
            if i not in selected:
                tag = f"[PARA_{i}] {p}"
                if total_chars + len(tag) > max_chars:
                    break
                parts.append(tag)
                total_chars += len(tag)

    return "\n\n".join(parts)


# ── Default strategy ──

_DEFAULT_KEYWORDS = [
    "supplier", "customer", "supply chain", "supply agreement",
    "raw material", "sole source", "single source", "dependent",
    "procurement", "distributor", "contract manufacturer",
    "critical mineral", "supply risk", "concentration risk",
]


def _default_strategy(paragraphs: List[str], **kwargs) -> str:
    return _build_from_keywords(paragraphs, _DEFAULT_KEYWORDS, **kwargs)


register_strategy("default", _default_strategy)


# ── USGS country/mineral strategy ──

_USGS_KEYWORDS = [
    "production", "producer", "mine", "mining", "refinery", "refining",
    "reserves", "resources", "export", "import", "trade",
    "metric ton", "tonnes", "percent", "world",
    "china", "congo", "australia", "chile", "indonesia", "brazil",
    "critical mineral", "strategic", "national defense",
    "supply", "demand", "consumption", "substitution",
    "stockpile", "recycling", "price",
]


def _usgs_strategy(paragraphs: List[str], **kwargs) -> str:
    extra_kw = kwargs.pop("extra_keywords", [])
    keywords = _USGS_KEYWORDS + extra_kw
    return _build_from_keywords(
        paragraphs, keywords,
        max_chars=kwargs.get("max_chars", 120_000),
        window=kwargs.get("window", 3),
    )


register_strategy("usgs_country_mineral", _usgs_strategy)


# ── SEC minerals strategy ──

_SEC_MINERALS_KEYWORDS = [
    "cobalt", "lithium", "nickel", "graphite", "rare earth",
    "copper", "manganese", "gallium", "germanium", "silicon",
    "tungsten", "titanium", "vanadium", "platinum", "tin",
    "zinc", "tantalum", "niobium", "chromium", "molybdenum",
    "antimony", "magnesium", "beryllium",
    "supplier", "supply chain", "raw material", "procurement",
    "sole source", "single source", "concentration risk",
    "mine", "mining", "refinery", "smelter", "processing",
    "battery", "cathode", "anode", "electrolyte", "cell",
    "conflict mineral", "responsible sourcing", "DRC",
    "export control", "trade restriction", "geopolitical",
]


def _sec_minerals_strategy(paragraphs: List[str], **kwargs) -> str:
    extra_kw = kwargs.pop("extra_keywords", [])
    keywords = _SEC_MINERALS_KEYWORDS + extra_kw
    return _build_from_keywords(
        paragraphs, keywords,
        max_chars=kwargs.get("max_chars", 120_000),
    )


register_strategy("sec_minerals", _sec_minerals_strategy)


# ── SEC generic strategy ──

_SEC_GENERIC_KEYWORDS = [
    "supplier", "customer", "revenue", "supply chain",
    "supply agreement", "purchase commitment", "long-term agreement",
    "sole source", "single source", "concentration risk",
    "largest customer", "major customer", "significant customer",
    "contract manufacturer", "distributor", "reseller",
    "outsource", "subcontract", "third party",
]


def _sec_generic_strategy(paragraphs: List[str], **kwargs) -> str:
    return _build_from_keywords(paragraphs, _SEC_GENERIC_KEYWORDS, **kwargs)


register_strategy("sec_generic", _sec_generic_strategy)

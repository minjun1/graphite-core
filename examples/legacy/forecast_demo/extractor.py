#!/usr/bin/env python3
"""
Hurricane Beryl (2024) extractor — builds infrastructure dependency edges
from real Beryl reports using structural knowledge.

Adds Freeport LNG and adjusts edges for the 2024 event context.
"""

from typing import List

from graphite import ExtractedEdge, NodeRef, Provenance
from graphite.enums import EdgeType, SourceType, ConfidenceLevel, AssertionMode


def _edge(from_ref, to_ref, etype, quote, src_id, date, **kw):
    """Shorthand for creating an ExtractedEdge with Beryl provenance."""
    return ExtractedEdge(
        from_node=from_ref,
        to_node=to_ref,
        edge_type=etype,
        assertion_mode=AssertionMode.EXTRACTED,
        provenance=[
            Provenance(
                source_id=src_id,
                source_type=SourceType.PUBLIC_REPORT,
                evidence_quote=quote,
                confidence=ConfidenceLevel.HIGH,
                observed_at=date,
                valid_from=kw.get("valid_from", date),
                valid_to=kw.get("valid_to", "2024-07-15"),
                snapshot_id="beryl-2024",
            )
        ],
    )


def get_beryl_edges() -> List[ExtractedEdge]:
    """Return all edges for the Beryl 2024 demo."""
    return [
        # ── DEPENDS_ON: facility → port ──
        _edge(
            NodeRef.facility("EXXON_BAYTOWN"),
            NodeRef.asset("PORT_HOUSTON"),
            EdgeType.DEPENDS_ON,
            "ExxonMobil's Baytown refinery complex reduced throughput on July 7, 2024, citing storm preparedness",
            "beryl-exxon-port",
            "2024-07-07",
        ),
        _edge(
            NodeRef.facility("VALERO_HOUSTON"),
            NodeRef.asset("PORT_HOUSTON"),
            EdgeType.DEPENDS_ON,
            "Valero Energy operations at its Houston plant were affected by post-storm power disruptions",
            "beryl-valero-port",
            "2024-07-08",
        ),
        _edge(
            NodeRef.facility("FREEPORT_LNG"),
            NodeRef.asset("PORT_HOUSTON"),
            EdgeType.DEPENDS_ON,
            "Freeport LNG saw disrupted operations and slowed production due to Hurricane Beryl",
            "beryl-freeport-port",
            "2024-07-08",
        ),
        # ── DEPENDS_ON: facility → grid ──
        _edge(
            NodeRef.facility("EXXON_BAYTOWN"),
            NodeRef.asset("CENTERPOINT_GRID"),
            EdgeType.DEPENDS_ON,
            "CenterPoint Energy reported widespread grid failures; over 2.7 million customers lost power",
            "beryl-exxon-grid",
            "2024-07-08",
        ),
        _edge(
            NodeRef.facility("FREEPORT_LNG"),
            NodeRef.asset("CENTERPOINT_GRID"),
            EdgeType.DEPENDS_ON,
            "The facility depends on pipeline deliveries of natural gas and stable electrical power, both affected by the storm",
            "beryl-freeport-grid",
            "2024-07-08",
        ),
        _edge(
            NodeRef.facility("VALERO_HOUSTON"),
            NodeRef.asset("CENTERPOINT_GRID"),
            EdgeType.DEPENDS_ON,
            "Valero Energy operations affected by post-storm power disruptions from CenterPoint grid failure",
            "beryl-valero-grid",
            "2024-07-08",
        ),
        # ── LOCATED_IN: asset → region ──
        _edge(
            NodeRef.asset("PORT_HOUSTON"),
            NodeRef.region("HOUSTON_METRO"),
            EdgeType.LOCATED_IN,
            "Port of Houston suspended all operations on July 7, 2024",
            "beryl-port-location",
            "2024-07-07",
        ),
        _edge(
            NodeRef.asset("CENTERPOINT_GRID"),
            NodeRef.region("HOUSTON_METRO"),
            EdgeType.LOCATED_IN,
            "CenterPoint Energy, the electric utility serving Houston metropolitan area",
            "beryl-grid-location",
            "2024-07-08",
        ),
        # ── RISK_FLOWS_TO: port → ship channel ──
        _edge(
            NodeRef.asset("PORT_HOUSTON"),
            NodeRef.corridor("HOUSTON_SHIP_CHANNEL"),
            EdgeType.RISK_FLOWS_TO,
            "Houston Ship Channel experienced transit restrictions before halting all traffic",
            "beryl-port-channel",
            "2024-07-07",
        ),
    ]

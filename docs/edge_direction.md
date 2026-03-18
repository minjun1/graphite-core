# Edge Direction Convention

## Principle

All edges in a Graphite graph use **semantic (real-world) direction** — the arrow points in the direction that makes sense when you read the relationship aloud.

## Examples

| Edge | Direction | Read as |
| --- | --- | --- |
| `facility:REFINERY_A → asset:PORT_HOUSTON` | facility → asset | "Refinery A **depends on** Port of Houston" |
| `asset:PORT_HOUSTON → region:US-TX-HOUSTON` | asset → region | "Port of Houston is **located in** Houston" |
| `region:US-TX-HOUSTON → region:US-TX-GALVESTON` | region → region | "Houston is **adjacent to** Galveston" |
| `country:CD → mineral:COBALT` | country → mineral | "DRC **produces** Cobalt" |
| `mineral:COBALT → company:CATL` | mineral → company | "Cobalt is **used by** CATL" |

### DEPENDS_ON Convention

**`A DEPENDS_ON B` = A cannot operate normally without B.**

Examples:
- `facility:DATA_CENTER DEPENDS_ON facility:SUBSTATION` — data center needs power
- `facility:REFINERY DEPENDS_ON asset:PORT` — refinery needs crude oil deliveries via port
- `facility:WAREHOUSE DEPENDS_ON corridor:I10_HIGHWAY` — warehouse needs road access

## Propagation Direction

The propagation engine (`simulate.py`) handles traversal direction **separately** from semantic direction:

- `is_supply=True` → follows `G.successors()` (forward along edges)
- `is_supply=False` → the engine can be extended to follow `G.predecessors()` (reverse)

**Do NOT flip edge direction to make propagation easier.** The graph should always reflect real-world semantics. The transmission function handles directionality.

## Geo-Climate Risk Propagation

For geo-climate scenarios, risk typically flows:

```
[Shock Event] → Region → Asset/Facility → Downstream Facility → Company
```

Since edges like `LOCATED_IN` point *from* asset *to* region, the scenario runner injects shocks at region nodes and propagates forward through `EXPOSED_TO`/`RISK_FLOWS_TO` edges, which naturally point in the risk-flow direction.

When semantic direction doesn't align with risk flow (e.g., `LOCATED_IN`), the alpha function can handle this by treating reverse-traversal edges differently.

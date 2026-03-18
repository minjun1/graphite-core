# Node Taxonomy

## Core Node Types

Node types are defined in `graphite.enums.NodeType` and used through `NodeRef` factory methods.

### Financial / Supply Chain (v0.1)

| Type | Definition | NodeRef Factory | Examples |
|---|---|---|---|
| `COMPANY` | Legal entity with a ticker or identifier | `NodeRef.company("TSLA")` | Tesla, CATL, Glencore |
| `COUNTRY` | Nation-state, identified by ISO code | `NodeRef.country("CD")` | DRC, China, Australia |
| `MINERAL` | Raw material or commodity | `NodeRef.mineral("COBALT")` | Cobalt, Lithium, Graphite |

### Geo-Climate / Physical Infrastructure (v0.2)

| Type | Definition | NodeRef Factory | Examples |
|---|---|---|---|
| `REGION` | Geographic aggregation unit | `NodeRef.region("US-TX-HOUSTON")` | Houston Metro, Gulf Coast, FEMA Zone AE |
| `ASSET` | Owned/operated economic or infrastructure object | `NodeRef.asset("PORT_HOUSTON")` | Port of Houston, Substation #42, Pipeline segment |
| `FACILITY` | Site-bound operational unit | `NodeRef.facility("REFINERY_A")` | Motiva refinery, AWS data center, Warehouse |
| `CORRIDOR` | Network/pathway connecting regions or assets | `NodeRef.corridor("HOUSTON_SHIP_CHANNEL")` | I-10 highway, Houston Ship Channel |

## Boundary Cases

| Entity | Recommended Type | Reasoning |
|---|---|---|
| Port | `ASSET` | Economic infrastructure object, not site-bound to a single building |
| Refinery | `FACILITY` | Site-bound operational unit |
| Substation | `FACILITY` | Site-bound operational unit |
| Pipeline segment | `CORRIDOR` | Network pathway object |
| Highway | `CORRIDOR` | Network pathway object |
| FEMA flood zone | `REGION` | Geographic aggregation unit |

### FACILITY vs ASSET

`FACILITY` is an operationally site-bound asset category. Core treats them as separate `NodeType` values, but in domain semantics `FACILITY` is a subtype of `ASSET`. Use `ASSET` for infrastructure objects that span multiple sites or are not site-bound (e.g., a port authority); use `FACILITY` for single-site operational units (e.g., a refinery, data center).

## Temporal Fields

All temporal string fields (`observed_at`, `valid_from`, `valid_to` on `Provenance`) **must use ISO 8601 format**:

- Date only: `2017-08-25`
- Date + time: `2017-08-25T18:00:00Z`
- With timezone: `2017-08-25T13:00:00-05:00`

## Geometry

Graphite core is **geometry-agnostic**. Spatial attributes (lat/lon, bbox, geometry_ref) live in domain-specific edge/node attributes, not in core schemas. This keeps core lightweight and avoids coupling to GIS libraries.

Domain packs may add geometry via `ExtractedEdge.attributes`:
```python
attributes={"lat": 29.76, "lon": -95.36, "bbox": [...]}
```

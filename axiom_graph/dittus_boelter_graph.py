"""Dittus-Boelter convective heat transfer axiom graph.

Three physical assumptions underpin Nu = 0.023 Re^0.8 Pr^0.4:

    A1_turbulent_flow    -- Re > 10000 (fully turbulent pipe flow)
    A2_moderate_prandtl  -- 0.6 < Pr < 160 (boundary-layer similarity)
    A3_developed_flow    -- L/D > 10 (thermally and hydrodynamically developed)

Two derived quantities, both requiring all three assumptions:

    D1_nusselt_number  <- A1, A2, A3
    D2_heat_flux       <- A1, A2, A3  (via D1)

Provenance limitation: unlike linear elasticity where D3 has only two
parents (enabling the system to trust D3 even when A3/A4 fire), here
every derived result traces back to all three assumptions simultaneously.
Provenance will always flag D1 and D2 together -- it cannot isolate which
assumption is responsible. The interesting result in this domain is
therefore predicate behavior (non-integer Re^0.8 weights, U-shaped Pr
boundary) rather than differential provenance.
"""

from __future__ import annotations

from axiom_graph.edges import DerivationEdge
from axiom_graph.graph import AxiomGraph
from axiom_graph.nodes import Node


def build_db_graph() -> AxiomGraph:
    """Construct and return the Dittus-Boelter axiom DAG."""
    g = AxiomGraph()

    # ---- Assumption nodes ------------------------------------------------
    g.add_node(Node(
        id="A1_turbulent_flow", kind="assumption",
        label="Turbulent flow (A1)",
        description=(
            "Reynolds number Re = rho*v*D/mu exceeds ~10000, ensuring "
            "fully turbulent pipe flow. Dittus-Boelter was derived and "
            "validated in this regime; at lower Re (transitional or "
            "laminar) the correlation significantly over- or under-predicts Nu."
        ),
    ))
    g.add_node(Node(
        id="A2_moderate_prandtl", kind="assumption",
        label="Moderate Prandtl number (A2)",
        description=(
            "Prandtl number Pr = mu*cp/k lies in [0.6, 160]. "
            "The boundary-layer analogy that underpins Dittus-Boelter "
            "assumes the thermal and velocity boundary layers are of "
            "comparable thickness. Liquid metals (Pr << 1) and very "
            "viscous oils (Pr >> 160) violate this assumption. "
            "This gives a U-shaped validity region -- Pr can be too low "
            "OR too high -- which is a new challenge for monotone predictors."
        ),
    ))
    g.add_node(Node(
        id="A3_developed_flow", kind="assumption",
        label="Thermally developed flow (A3)",
        description=(
            "Pipe length-to-diameter ratio L/D exceeds ~10, ensuring that "
            "both the hydrodynamic and thermal boundary layers are fully "
            "developed. In short pipes (L/D < 10) entrance effects add a "
            "~(D/L)*C correction to Nu that the Dittus-Boelter equation "
            "ignores, causing under-prediction of the actual heat transfer."
        ),
    ))

    # ---- Derived nodes ---------------------------------------------------
    g.add_node(Node(
        id="D1_nusselt_number", kind="derived",
        label="Nusselt number  Nu = 0.023 Re^0.8 Pr^0.4",
        description=(
            "Dimensionless convective heat transfer coefficient. "
            "Requires turbulent flow (A1) for the Re^0.8 scaling, "
            "moderate Pr (A2) for the Pr^0.4 scaling and boundary-layer "
            "analogy, and developed flow (A3) so that entrance effects "
            "are negligible. All three assumptions must hold simultaneously."
        ),
    ))
    g.add_node(Node(
        id="D2_heat_flux", kind="derived",
        label="Convective heat flux  q = (Nu*k/D) * delta_T",
        description=(
            "Dimensional heat flux from pipe wall to fluid, obtained by "
            "converting Nu to the heat transfer coefficient h = Nu*k/D "
            "and applying Newton's law of cooling: q = h * delta_T. "
            "Inherits all three assumptions from D1 (Nusselt number)."
        ),
    ))

    # ---- Edges -----------------------------------------------------------
    g.add_edge(DerivationEdge(
        id="DB_E1",
        premise_ids=["A1_turbulent_flow", "A2_moderate_prandtl", "A3_developed_flow"],
        conclusion_id="D1_nusselt_number",
        rule_label="Turbulent + moderate Pr + developed -> Nu = 0.023 Re^0.8 Pr^0.4",
        description=(
            "A1 (Re > 10000) activates the turbulent power-law Re^0.8 scaling. "
            "A2 (0.6 < Pr < 160) validates the Pr^0.4 factor from the "
            "boundary-layer Reynolds analogy. "
            "A3 (L/D > 10) ensures the bulk of the pipe is in the fully "
            "developed regime so the Dittus-Boelter constant 0.023 applies."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="DB_E2",
        premise_ids=["D1_nusselt_number"],
        conclusion_id="D2_heat_flux",
        rule_label="Nu -> h = Nu*k/D -> q = h*delta_T",
        description=(
            "The convective heat transfer coefficient h = Nu * k / D is "
            "obtained by dimensional analysis. Newton's law of cooling then "
            "gives the wall heat flux q = h * (T_wall - T_bulk). "
            "D2 inherits all three assumption dependencies through D1."
        ),
    ))

    return g


if __name__ == "__main__":
    g = build_db_graph()

    # 1. Print all nodes
    print("=== Nodes ===")
    for node in g.nodes.values():
        print(f"  [{node.kind:10s}]  {node.id}  --  {node.label}")

    # 2. Print provenance footprints for derived nodes
    print("\n=== Provenance footprints ===")
    derived_nodes = [n for n in g.nodes.values() if n.kind == "derived"]
    footprints: dict[str, frozenset[str]] = {}
    for node in derived_nodes:
        fp = g.ancestor_assumptions(node.id)
        footprints[node.id] = fp
        print(f"  {node.id}: {sorted(fp)}")

    # 3. Assertions
    expected = frozenset({"A1_turbulent_flow", "A2_moderate_prandtl", "A3_developed_flow"})

    assert footprints["D1_nusselt_number"] == expected, (
        f"D1 footprint mismatch!\n"
        f"  expected: {sorted(expected)}\n"
        f"  got:      {sorted(footprints['D1_nusselt_number'])}"
    )
    assert footprints["D2_heat_flux"] == expected, (
        f"D2 footprint mismatch!\n"
        f"  expected: {sorted(expected)}\n"
        f"  got:      {sorted(footprints['D2_heat_flux'])}"
    )

    print("\nAll assertions passed.")
    print()
    print("Provenance note: D1 and D2 always share the same footprint {A1, A2, A3}.")
    print("Provenance cannot discriminate which assumption fired -- all are flagged")
    print("together. The interesting results for this domain are in predicate behavior:")
    print("  A1: non-integer Re^0.8 weights (skip learns ~0.8 on all four raw inputs)")
    print("  A2: U-shaped Pr boundary (skip is monotone; MLP must compensate)")
    print("  A3: integer L/D weights (skip learns +1 on L, -1 on D exactly)")

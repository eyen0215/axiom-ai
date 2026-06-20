"""Pipe flow axiom graph.

Three physical assumptions underpin the Hagen-Poiseuille pipe flow model:

    A1_laminar_flow     — Re = rho*v*D/mu < 2300
    A2_fully_developed  — x > 0.06*Re*D  (past entrance length)
    A3_incompressible   — Ma = v/c_sound < 0.3

Two derived quantities with different provenance footprints:

    D1_velocity_profile  ← A1, A2          (parabolic HP profile)
    D2_pressure_drop     ← A1, A2, A3      (linear dP/dx = -128*mu*Q/(pi*D^4))

D2 is built on D1: once the fully-developed parabolic profile exists, the
pressure drop follows from integrating the momentum equation — but that
integration additionally requires the flow to be incompressible (A3) so
that density drops out of the continuity equation.
"""

from __future__ import annotations

from axiom_graph.edges import DerivationEdge
from axiom_graph.graph import AxiomGraph
from axiom_graph.nodes import Node


def build_pipe_graph() -> AxiomGraph:
    """Construct and return the pipe flow axiom DAG."""
    g = AxiomGraph()

    # ---- Assumption nodes ------------------------------------------------
    g.add_node(Node(
        id="A1_laminar_flow", kind="assumption",
        label="Laminar flow (A1)",
        description=(
            "Reynolds number Re = rho*v*D/mu remains below 2300. "
            "Violated when flow transitions to turbulent, breaking the "
            "parabolic velocity profile and the Hagen-Poiseuille derivation."
        ),
    ))
    g.add_node(Node(
        id="A2_fully_developed", kind="assumption",
        label="Fully developed (A2)",
        description=(
            "Downstream position x exceeds the entrance length L_entry = 0.06*Re*D. "
            "Violated near the pipe inlet where the boundary layer is still growing "
            "and the velocity profile has not yet reached its parabolic shape."
        ),
    ))
    g.add_node(Node(
        id="A3_incompressible", kind="assumption",
        label="Incompressible (A3)",
        description=(
            "Mach number Ma = v/c_sound remains below 0.3. "
            "Violated at high flow speeds where density variations become "
            "significant and the incompressible continuity equation breaks down."
        ),
    ))

    # ---- Derived nodes ---------------------------------------------------
    g.add_node(Node(
        id="D1_velocity_profile", kind="derived",
        label="Parabolic velocity profile",
        description=(
            "The Hagen-Poiseuille parabolic profile u(r) = u_max*(1 - (r/R)^2). "
            "Requires laminar flow (A1) so the Navier-Stokes equation linearises, "
            "and fully-developed conditions (A2) so the radial profile is constant "
            "along x and inertial entrance effects have decayed."
        ),
    ))
    g.add_node(Node(
        id="D2_pressure_drop", kind="derived",
        label="Pressure drop  dP/dx",
        description=(
            "Linear pressure drop dP/dx = -128*mu*Q / (pi*D^4)  (Hagen-Poiseuille). "
            "Derived by integrating the momentum equation over the parabolic profile "
            "(D1); additionally requires incompressibility (A3) so that density "
            "is constant and drops out of the continuity equation."
        ),
    ))

    # ---- Edges -----------------------------------------------------------
    g.add_edge(DerivationEdge(
        id="PE1",
        premise_ids=["A1_laminar_flow", "A2_fully_developed"],
        conclusion_id="D1_velocity_profile",
        rule_label="Laminar + fully developed → parabolic u(r)",
        description=(
            "A1 (Re < 2300) ensures the viscous term dominates so the "
            "Navier-Stokes equation reduces to the linear Stokes equation. "
            "A2 (x > L_entry) ensures the boundary-layer transient has decayed "
            "and the radial profile is self-similar along the pipe axis."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="PE2",
        premise_ids=["D1_velocity_profile", "A3_incompressible"],
        conclusion_id="D2_pressure_drop",
        rule_label="Parabolic profile + incompressible → dP/dx = -128μQ/(πD⁴)",
        description=(
            "Integrating the axial momentum equation over the parabolic profile "
            "(D1) gives the volumetric flow rate Q = pi*R^4*(−dP/dx)/(8*mu). "
            "A3 (Ma < 0.3) ensures rho is constant, so the continuity equation "
            "reduces to div(u) = 0 and density drops out of the momentum balance."
        ),
    ))

    return g


if __name__ == "__main__":
    g = build_pipe_graph()

    print("=== Nodes ===")
    for node in g.nodes.values():
        print(f"  [{node.kind:10s}]  {node.id}  --  {node.label}")

    print("\n=== Provenance footprints (derived nodes) ===")
    footprints: dict[str, frozenset[str]] = {}
    for node in g.nodes.values():
        if node.kind == "derived":
            fp = g.ancestor_assumptions(node.id)
            footprints[node.id] = fp
            print(f"  {node.id}: {sorted(fp)}")

    expected_d1 = frozenset({"A1_laminar_flow", "A2_fully_developed"})
    expected_d2 = frozenset({"A1_laminar_flow", "A2_fully_developed", "A3_incompressible"})

    assert footprints["D1_velocity_profile"] == expected_d1, (
        f"D1 footprint mismatch!\n"
        f"  expected: {sorted(expected_d1)}\n"
        f"  got:      {sorted(footprints['D1_velocity_profile'])}"
    )
    assert footprints["D2_pressure_drop"] == expected_d2, (
        f"D2 footprint mismatch!\n"
        f"  expected: {sorted(expected_d2)}\n"
        f"  got:      {sorted(footprints['D2_pressure_drop'])}"
    )

    print("\nAll assertions passed.")

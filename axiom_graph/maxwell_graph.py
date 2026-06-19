"""Maxwell's equations axiom graph in linear dielectric media.

Four physical assumptions:
    A1_linear_media  — D = epsilon*E (linear polarisation holds)
    A2_quasi_static  — displacement current negligible vs conduction current
    A3_homogeneous   — permittivity epsilon is spatially uniform
    A4_isotropic     — permittivity epsilon is a scalar, not a tensor

Four derived quantities with different provenance footprints:
    D1 wave_speed     <- A1, A2          (propagation + linear material)
    D2 impedance      <- A1, A3, A4      (material ratio; frequency-independent)
    D3 energy_density <- A1              (local scalar; only linearity needed)
    D4 polarization   <- A1, A4          (D || E requires scalar epsilon)

A2 appears ONLY in D1's footprint — the load-bearing fact for Scenario B:
when A2 fires at high frequency, only D1 becomes SUSPECT while D2/D3/D4
remain TRUSTED.  Directly mirrors LE Scenario C (A5 fires -> only D4 SUSPECT).
"""

from __future__ import annotations

from axiom_graph.edges import DerivationEdge
from axiom_graph.graph import AxiomGraph
from axiom_graph.nodes import Node


def build_maxwell_graph() -> AxiomGraph:
    """Construct and return the Maxwell equations axiom DAG."""
    g = AxiomGraph()

    # ---- Assumption nodes -----------------------------------------------
    g.add_node(Node(
        id="A1_linear_media", kind="assumption",
        label="Linear media (A1)",
        description=(
            "D = epsilon*E holds (linear polarisation). "
            "Violated when E exceeds E_sat (~1e8 V/m) and nonlinear "
            "saturation effects dominate the polarisation response."
        ),
    ))
    g.add_node(Node(
        id="A2_quasi_static", kind="assumption",
        label="Quasi-static (A2)",
        description=(
            "Displacement current omega*epsilon is negligible compared to "
            "conduction current sigma_eff.  Threshold: omega*epsilon/sigma_eff < 0.01. "
            "Violated above ~79.9 kHz for the lossy glass model "
            "(sigma_eff=1e-3 S/m, epsilon=1.992e-11 F/m)."
        ),
    ))
    g.add_node(Node(
        id="A3_homogeneous", kind="assumption",
        label="Homogeneous (A3)",
        description=(
            "Permittivity epsilon is spatially uniform (CV < 10%). "
            "Violated in graded or inhomogeneous dielectric structures."
        ),
    ))
    g.add_node(Node(
        id="A4_isotropic", kind="assumption",
        label="Isotropic (A4)",
        description=(
            "Permittivity epsilon is a scalar, not a tensor "
            "(|epsilon_perp/epsilon_parallel - 1| < 0.05). "
            "Violated in birefringent crystals and anisotropic media."
        ),
    ))

    # ---- Derived nodes --------------------------------------------------
    g.add_node(Node(
        id="D1_wave_speed", kind="derived",
        label="Wave speed v",
        description=(
            "Phase velocity v = 1/sqrt(epsilon*mu). Requires linear media (A1) "
            "and the quasi-static limit where the simple non-dispersive "
            "dispersion relation holds without retardation corrections (A2)."
        ),
    ))
    g.add_node(Node(
        id="D2_impedance", kind="derived",
        label="Impedance eta",
        description=(
            "Wave impedance eta = sqrt(mu/epsilon). Material property ratio only; "
            "requires linear epsilon (A1), spatially uniform epsilon (A3), and "
            "scalar epsilon so mu/epsilon is a single number (A4). "
            "Frequency-independent; A2 not required."
        ),
    ))
    g.add_node(Node(
        id="D3_energy_density", kind="derived",
        label="Energy density u",
        description=(
            "Electromagnetic energy density u = 1/2*epsilon*E^2 + 1/2*mu*H^2. "
            "Local scalar formula valid pointwise; requires only linear "
            "constitutive relation (A1). Homogeneity and isotropy not needed."
        ),
    ))
    g.add_node(Node(
        id="D4_polarization", kind="derived",
        label="Polarization direction",
        description=(
            "Fixed linear polarization requires D || E, which holds when "
            "D = epsilon*E (A1) with scalar epsilon so the polarization "
            "direction of D matches E (A4). A2 and A3 not required."
        ),
    ))

    # ---- Edges ----------------------------------------------------------
    g.add_edge(DerivationEdge(
        id="MW1",
        premise_ids=["A1_linear_media", "A2_quasi_static"],
        conclusion_id="D1_wave_speed",
        rule_label="A1+A2 -> v = 1/sqrt(epsilon*mu)",
        description=(
            "Simple wave-speed formula valid in linear non-dispersive media (A1) "
            "under the quasi-static approximation (A2), where retardation and "
            "frequency-dependent dispersion corrections are negligible."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="MW2",
        premise_ids=["A1_linear_media", "A3_homogeneous", "A4_isotropic"],
        conclusion_id="D2_impedance",
        rule_label="A1+A3+A4 -> eta = sqrt(mu/epsilon)",
        description=(
            "Impedance depends only on material properties. Requires linear "
            "epsilon (A1), spatially uniform epsilon (A3), and scalar epsilon "
            "so the ratio mu/epsilon is a single well-defined constant (A4)."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="MW3",
        premise_ids=["A1_linear_media"],
        conclusion_id="D3_energy_density",
        rule_label="A1 -> u = 1/2*epsilon*E^2 + 1/2*mu*H^2",
        description=(
            "Local energy density formula requires only linear constitutive "
            "relation D = epsilon*E (A1). Holds pointwise regardless of spatial "
            "uniformity (A3) or tensor structure (A4)."
        ),
    ))
    g.add_edge(DerivationEdge(
        id="MW4",
        premise_ids=["A1_linear_media", "A4_isotropic"],
        conclusion_id="D4_polarization",
        rule_label="A1+A4 -> D || E",
        description=(
            "Fixed polarization direction requires D = epsilon*E (A1) with "
            "scalar epsilon so the polarization direction of D is parallel "
            "to E (A4). A2 (quasi-static) and A3 (homogeneous) not required."
        ),
    ))

    return g


if __name__ == "__main__":
    g = build_maxwell_graph()

    print("=== Nodes ===")
    for node in g.nodes.values():
        print(f"  [{node.kind:10s}]  {node.id}  --  {node.label}")

    print("\n=== Provenance footprints (DerivedNodes) ===")
    derived_nodes = [n for n in g.nodes.values() if n.kind == "derived"]
    footprints: dict[str, frozenset[str]] = {}
    for node in derived_nodes:
        fp = g.ancestor_assumptions(node.id)
        footprints[node.id] = fp
        print(f"  {node.id}: {sorted(fp)}")

    # Assertions specified in CLAUDE_MAXWELL.md
    expected_d1 = frozenset({"A1_linear_media", "A2_quasi_static"})
    expected_d3 = frozenset({"A1_linear_media"})

    assert footprints["D1_wave_speed"] == expected_d1, (
        f"D1 footprint mismatch!\n"
        f"  expected: {sorted(expected_d1)}\n"
        f"  got:      {sorted(footprints['D1_wave_speed'])}"
    )
    assert footprints["D3_energy_density"] == expected_d3, (
        f"D3 footprint mismatch!\n"
        f"  expected: {sorted(expected_d3)}\n"
        f"  got:      {sorted(footprints['D3_energy_density'])}"
    )

    print("\nAll assertions passed.")

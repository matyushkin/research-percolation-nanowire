# Open Research Problems: Percolation in Nanowire Networks

Priority ranking for paper ideas. Updated: March 2026.

---

## Paper Idea #1: Bridge Percolation with Random Crack Topology ⭐⭐⭐

**Gap:** Baret et al. (2024) studied bridge percolation only for parallel straight cracks.
Real cracked ITO films have random crack networks (branching, varying width, hierarchical).

**What to do:**
- Generate random crack networks: Poisson-Voronoi tessellation, fractal crack patterns
- Place AgNW randomly, count fraction bridging cracks
- Find effective percolation threshold N_bridge(crack topology)
- Compare with parallel-crack model and experiment

**Expected result:** N_bridge depends strongly on crack topology; Voronoi networks require fewer wires than parallel cracks due to shorter mean crack segment length.

**Novelty:** Direct computational follow-up to the only existing bridge percolation paper.
**Target:** Physical Review E or Nanoscale
**Code needed:** Voronoi tessellation (scipy.spatial), segment intersection, Union-Find

---

## Paper Idea #2: AgNW + Quantum Dots Hybrid Network ⭐⭐⭐

**Gap:** Nobody has modeled the combined optical/electrical properties of AgNW networks
functionalized with quantum dots near the percolation threshold.

**What to do:**
- Simulate AgNW network near η_c
- Place QDs at wire intersections (junction sites) or along wires
- Calculate: (a) electrical conductivity via Kirchhoff, (b) optical near-field enhancement at junctions
- Study how FRET efficiency or plasmon-exciton coupling depends on η

**Expected result:** Near η_c, near-field hotspots percolate → dramatic enhancement of QD emission.
Connection to author's paper: Enhanced Luminescence of QD near Ag/SiO₂ NPs (Tech. Phys. Lett. 2018, 19 cit.)

**Novelty:** Unique intersection of author's QD + plasmonic expertise with percolation theory.
**Target:** ACS Nano or Nanoscale
**Code needed:** AgNW network + dipole coupling model (analytical Mie + near-field)

---

## Paper Idea #3: IR Optical Properties at Percolation Threshold ⭐⭐

**Gap:** AgNW network IR spectroscopy (mid-IR, 3–15 μm) across percolation transition not studied.
Author has direct expertise: IR scattering in AAO membranes (Inorganic Materials 2018).

**What to do:**
- Model AgNW network as collection of coupled antennas
- Calculate effective medium optical response (Bruggeman or coupled-dipole)
- Vary η across threshold, compute transmission/absorption spectra in mid-IR
- Find signature of percolation transition in optical spectra

**Expected result:** Anomalous IR absorption feature at η_c due to onset of long-range current paths.
**Target:** Physical Review B or Journal of Applied Physics
**Code needed:** Coupled-dipole model or transfer matrix for 2D antenna array

---

## Paper Idea #4: Systematic t(κ) Crossover Study ⭐⭐

**Gap:** Conductivity exponent t depends on κ = R_j/(R_w·L), but different groups report conflicting values.
No systematic MC study with wide κ range (10⁻³–10⁴) and controlled finite-size scaling.

**What to do:**
- Generate stick networks for η from 0.8·η_c to 3·η_c
- Vary κ over 7 decades
- Extract t via log-log fit σ vs (η - η_c)
- Careful finite-size scaling to remove boundary effects

**Expected result:** Smooth crossover curve t(κ) with two plateaus: t→1.30 (κ→0) and t→2.0 (κ→∞).
**Target:** Physical Review E
**Code needed:** Standard stick percolation + Kirchhoff solver (scipy.sparse)

---

## Implementation Order

1. **Start with Paper #4** — simplest code, validates the simulation framework
2. **Then Paper #1** — adds Voronoi geometry, novel result, fast to publish
3. **Then Paper #2** — requires optical model, most impactful
4. **Then Paper #3** — builds on #2, requires IR dielectric data for Ag

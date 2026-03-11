# Literature Notes: Percolation in Nanowire Networks

Collected during initial research scouting (March 2026).

---

## Key Papers

### Foundational Theory

| Paper | Key result | Notes |
|---|---|---|
| Balberg & Binenbaum, PRB (1983) | First MC for stick percolation | η_c ≈ 5.63 for thin sticks |
| Li & Zhang, PRE 80, 040104 (2009) | Precise η_c for sticks | Most cited reference for threshold |
| PRE 88, 032134 (2013) | Quasi-2D model for nanowire films | 3D crossover when h ~ L |
| PRE 81, 021120 (2010) | Conductivity exponents vs κ = R_j/R_w | t crossover from 1.3 to ~2 |

### Bridge Percolation (NEW, 2025)

- **Baret et al., Nanoscale (2024)**: "Bridge percolation: electrical connectivity of discontinued conducting slabs by metallic nanowires"
  - URL: https://pubs.rsc.org/en/content/articlelanding/2024/nr/d3nr05850f
  - Key result: sparse AgNW on cracked ITO restore conductivity with ~35x less material
  - Model: parallel straight cracks only — random crack topology NOT studied
  - Open: 3D geometry, random Voronoi cracks, polydisperse NW length, failure statistics

- **Bulletin SRSL (2025)**: Extended version from Université de Liège
  - URL: https://popups.uliege.be/0037-9565/index.php?id=12531

### Nanowire Network Reviews & Recent Work

- **Advanced Electronic Materials (2025)**: ZnO nanowire percolation — shortest-path universalization approach to close theory-experiment gap
  - URL: https://advanced.onlinelibrary.wiley.com/doi/10.1002/aelm.202500242
  - Key finding: standard stick model underestimates real threshold

- **Taylor & Francis review (2025)**: "Recent advances in metallic nanowire based transparent electrodes"
  - URL: https://www.tandfonline.com/doi/full/10.1080/23746149.2025.2573818

- **Nanoscale, RSC (2026)**: Optical-electrical coupling in AgNW networks (stretchable electrodes)
  - URL: https://pubs.rsc.org/en/content/articlelanding/2026/nr/d5nr04505c

- **Nanoscale Horizons (2018)**: Comparison MC vs experiment — key reference for gap between theory and reality
  - URL: https://pubs.rsc.org/en/content/articlelanding/2018/nh/c8nh00066b

### Graph Theory Applied to Nanowire Networks

- **Sannicolo et al. (2022)**: De-densifying AgNW networks using betweenness centrality
  - URL: https://www.sciencedirect.com/science/article/pii/S0927025622004116
  - Key idea: betweenness centrality identifies current bottlenecks → targeted wire removal

- **ScienceDirect (2024)**: Layer-by-layer assembled NW networks, graph-theoretical multifunctional coatings
  - URL: https://www.sciencedirect.com/science/article/pii/S2590238524004934

### Monte Carlo for CNT/Polymer Composites

- **Composites Part A (2025)**: 3D MC + resistor network for CNT-GNP polymer composites
  - Key: 50-replicate MC, 3D RVE approach — current standard

- **MDPI Macromol (2025)**: Orientation-dependent percolation in CNT composites
  - Key finding: maximum connectivity at 55–60° mean angle (not 45° as assumed)
  - Open: full phase diagram vs orientation distribution

---

## Open Problems (Confirmed by Literature Search)

### HIGH PRIORITY (no papers found = open niches)

1. **AgNW + Quantum Dots hybrid network** — optical coupling at percolation threshold
   - Nobody studied how QD fluorescence/exciton coupling scales with NW percolation
   - Direct connection to author's expertise (QD + plasmonics + AgNW)
   - Status: **OPEN**

2. **Bridge percolation + random crack topology**
   - Baret 2024 used parallel straight cracks only
   - Voronoi/fractal random crack networks: **NOT STUDIED**
   - Status: **OPEN**

3. **AgNW network IR spectroscopy at percolation threshold**
   - How plasmonic resonances evolve across η_c in mid-IR: **NOT STUDIED**
   - Direct connection to author's AAO IR scattering papers
   - Status: **OPEN**

### PARTIALLY OPEN

4. **t(κ) crossover**: theory done (PRE 2010), but conflicting data between groups; systematic high-κ study missing

5. **Curvature + polydispersity combined effect**: studied separately, not together

### OCCUPIED (avoid)

6. Basic η_c for straight monodisperse sticks — done since 1983
7. Basic Kirchhoff conductivity for 2D networks — many papers

---

## Open-Source Code

| Tool | Language | Description | URL |
|---|---|---|---|
| Random-NWNs | Python/NetworkX | NW generation, graph analysis, conductivity | https://github.com/marcus-k/Random-NWNs |
| pyperc | Python | Invasion percolation (Sandia) | https://github.com/sandialabs/pyperc |
| percolation-python | Python | 2D grid percolation | https://github.com/ussserrr/percolation-python |

---

## Key Physical Parameters

| Parameter | Symbol | Typical values |
|---|---|---|
| Wire length | L | 10–100 μm |
| Wire diameter | d | 20–200 nm |
| Aspect ratio | AR = L/d | 100–1000 |
| Surface density | N [wires/μm²] | 0.001–0.1 |
| Junction resistance | R_j | 1–100 Ω (after annealing ~1 Ω) |
| Wire resistance | R_w | ~10 Ω/μm for Ag |
| Dimensionless density | η = N·L² | threshold η_c ≈ 5.63 |
| Resistance ratio | κ = R_j/(R_w·L) | 1–10⁴ — determines exponent t |

---

## Critical Exponents (2D)

| Exponent | Symbol | Value | Meaning |
|---|---|---|---|
| Correlation length | ν | 4/3 ≈ 1.333 | ξ ~ |p-p_c|^{-ν} |
| Order parameter | β | 5/36 ≈ 0.139 | P∞ ~ (p-p_c)^β |
| Conductivity | t | 1.30 (low κ) → ~2 (high κ) | σ ~ (N-N_c)^t |

---

## Target Journals

| Journal | IF | Scope | APC |
|---|---|---|---|
| npj Computational Materials | 9.4 | Computational materials | Yes (waiver available) |
| Physical Review Letters | 8.1 | Fundamental physics | No |
| Physical Review Research | 3.9 | Open access APS | No |
| Advanced Electronic Materials | 5.5 | Electronic nanomaterials | Yes |
| Nanoscale (RSC) | 6.7 | Nanomaterials | Yes |
| Physical Review E | 2.4 | Statistical physics | No |

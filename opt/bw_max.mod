### 3d_memory :: phase-1 bandwidth-maximization MINLP  (homogeneous macro)
###
### FROZEN VALIDATION ORACLE. The live model is now the Pyomo port in
### opt/model.py, driven by opt/run_bw_max.py. This AMPL file is retained only
### so the port can be re-validated against it (objective + structural .nl
### diff); it is no longer on the solve path. Do not add new physics here.
###
### Maximize the internal array-supply bandwidth of a homogeneous monolithic-3D
### memory macro at fixed capacity C, footprint A, and layer budget L.
### Every constraint corresponds one-to-one to the converged kernel in
### docs/model.tex / docs/model_slide.tex.
###
### Formulation: full MINLP. Array geometry (N_BL, N_WL) and sensing margin are
### continuous decision variables; the per-array timing/volume laws are written
### as explicit (nonconvex) constraints. Products of variables are kept to
### degree 2 by naming each product as an auxiliary variable, so the model is a
### nonconvex MIQCP + two univariate function constraints (log, 1/margin^2),
### all of which Gurobi 13 (nonconvex=2) handles directly.
###
### Units: time [ns], length [um], area [um^2], volume [um^3], capacity [bit].

#### ---- problem inputs (given) --------------------------------------------
param C        > 0;            # target capacity                      [bit]
param A        > 0;            # footprint (area per layer)           [um^2]
param L        > 0, integer;   # layer budget                         [layers]
param t_layer  > 0;            # physical thickness per layer         [um]
param VolBudget := A * L * t_layer;   # total 3D volume packed against [um^3]

#### ---- technology / calibration coefficients ----------------------------
#### TODO(calibration): placeholder values; replace with a memory_model fit.
param k_dec     > 0;   # decoder delay per address bit                [ns]
# WL/BL develop split into a distributed wire self-RC term (quadratic in the
# cells along the line; set by the BEOL wire stack + node, so SHAREABLE across
# cell technologies) and a cell/access loading term (linear; storage/gate/junction
# cap driven through the access resistance, so PER-TECHNOLOGY).
param k_wire_WL > 0;   # WL wire self-RC coeff        (per column^2)   [ns]  shared
param k_cell_WL >= 0;  # WL cell/gate loading coeff   (per column)     [ns]  per-tech
param k_wire_BL > 0;   # BL wire self-RC coeff        (per row^2)      [ns]  shared
param k_cell_BL >= 0;  # BL cell/access loading coeff (per row)        [ns]  per-tech
param t_SA0     > 0;   # base sense time                              [ns]
param t_restore >= 0;  # restore time for a destructive read          [ns]
param destructive binary;   # 1 => fold restore into t_SA (DRAM-like)
param t_sw      >= 0;  # selector switching time                      [ns]
param v_cell    > 0;   # volume per stored bit                        [um^3/bit]
param v_sa0     > 0;   # sense-amp volume at margin = 1               [um^3]
param k_vdec    > 0;   # decoder volume per array cell (~ Vol_arr)    [um^3/cell]
param v_sel     > 0;   # selector volume per (share x sense-amp)      [um^3]

#### ---- bounds (finite bounds are required for nonconvex spatial B&B) -----
param NBL_min > 0; param NBL_max > 0;
param NWL_min > 0; param NWL_max > 0;
param margin_min > 0; param margin_max > 0;
param Nshare_min integer > 0; param Nshare_max integer > 0;
param Nindep_max > 0;
param BW_max > 0;

#### ---- decision variables ------------------------------------------------
var N_BL    >= NBL_min, <= NBL_max;             # bitlines  (array columns)
var N_WL    >= NWL_min, <= NWL_max;             # wordlines (array rows)
var margin  >= margin_min, <= margin_max;       # sensing margin
var b_acc   >= 1, <= NBL_max;                   # bits per access (sense width)
var N_share integer >= Nshare_min, <= Nshare_max;# arrays sharing one periph set
var N_indep >= 1, <= Nindep_max;                # independent peripheral sets
var t_cycle >= t_SA0 * 0.5;        # steady-state cycle time
var BW      >= 0, <= BW_max;                    # objective

#### ---- defined timing terms (no variable products; log is univariate) ----
var t_dec = k_dec * log(N_WL) / log(2);         # decode depth ~ log2(rows)
var t_WL  = k_wire_WL * N_BL^2 + k_cell_WL * N_BL;  # WL: wire self-RC (~L^2) + cell/gate load (~cells)
var t_BL  = k_wire_BL * N_WL^2 + k_cell_BL * N_WL;  # BL: wire self-RC (~L^2) + cell/access load (~cells)
var t_SA  = t_SA0 + destructive * t_restore;     # restore folded in iff destructive
var sum_dev = t_dec + t_WL + t_BL + t_SA + t_sw; # full serial develop latency

#### ---- auxiliary product variables (each defining eqn is degree <= 2) ----
var cells_arr     >= 0, <= NBL_max*NWL_max;                 # cells per array
var N_tot         >= 0, <= Nindep_max*Nshare_max;           # total arrays
var N_SA          >= 0, <= Nindep_max*NBL_max;              # total sense amps
var total_cells   >= 0, <= NBL_max*NWL_max*Nindep_max*Nshare_max;
var cells_x_indep >= 0, <= NBL_max*NWL_max*Nindep_max;
var inv_m2        >= 0, <= 1/margin_min^2;                  # 1/margin^2
var vsa_term      >= 0, <= (1/margin_min^2)*Nindep_max*NBL_max;
var sel_term      >= 0, <= Nshare_max*Nindep_max*NBL_max;

subject to def_cells_arr:     cells_arr     = N_BL * N_WL;
subject to def_Ntot:          N_tot         = N_indep * N_share;
subject to def_NSA:           N_SA          = N_indep * b_acc;   # = N_indep*bits/access
subject to def_total_cells:   total_cells   = cells_arr * N_tot;
subject to def_cells_x_indep: cells_x_indep = cells_arr * N_indep;
subject to def_inv_m2:        inv_m2        = 1 / margin^2;      # Pelgrom A_SA ~ 1/margin^2
subject to def_vsa_term:      vsa_term      = inv_m2 * N_SA;
subject to def_sel_term:      sel_term      = N_share * N_SA;

#### ---- kernel constraints -------------------------------------------------
# cycle time = throughput bound: max of the two un-hideable floors and the
# develop latency amortized over the shared arrays  (model.tex Eq. 1)
subject to cyc_dec: t_cycle >= t_dec;
subject to cyc_sa:  t_cycle >= t_SA + t_sw;
subject to cyc_dev: t_cycle * N_share >= sum_dev;

# bandwidth identity  BW = N_SA / t_cycle = (bits/access) * N_indep / t_cycle
subject to bw_def:  BW * t_cycle = N_SA;

# sense-width cap: cannot sense more bits than bitlines  (b_acc <= N_BL)
subject to width_cap: b_acc <= N_BL;

# capacity: total stored bits must meet the target
subject to capacity: total_cells >= C;

# volume packing budget  (model.tex Eq. 4, as fit-in-budget)
subject to volume:
  v_cell*total_cells + v_sa0*vsa_term + k_vdec*cells_x_indep + v_sel*sel_term
    <= VolBudget;

#### ---- objective ---------------------------------------------------------
maximize bandwidth: BW;

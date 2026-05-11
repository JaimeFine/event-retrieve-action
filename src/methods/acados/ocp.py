from __future__ import annotations

import numpy as np

from .model import build_point_mass_model


def build_point_mass_ocp(acados_template, ca, *, horizon: int, dt: float, max_speed: float, max_accel: float, safety_radius: float, max_obstacles: int):
    # This OCP is intentionally lightweight (point-mass dynamics) to keep solver
    # generation and online solve latency predictable for repeated benchmark rollouts.
    AcadosModel = acados_template.AcadosModel
    AcadosOcp = acados_template.AcadosOcp
    ACADOS_INFTY = acados_template.ACADOS_INFTY

    model_data = build_point_mass_model(ca)

    model = AcadosModel()
    model.name = model_data["name"]
    model.x = model_data["x"]
    model.u = model_data["u"]
    model.xdot = model_data["xdot"]
    model.f_expl_expr = model_data["f_expl"]
    model.f_impl_expr = model_data["f_impl"]

    obstacle_param = ca.SX.sym("obstacle_param", 3 * max_obstacles)
    # Obstacles are passed as stage parameters to avoid rebuilding the solver
    # when intruder sets change between timesteps.
    model.p = obstacle_param

    px, py, pz = model.x[0], model.x[1], model.x[2]
    h_terms = []
    for i in range(max_obstacles):
        ox = obstacle_param[3 * i + 0]
        oy = obstacle_param[3 * i + 1]
        oz = obstacle_param[3 * i + 2]
        dist_sq = (px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2
        # h(x,p) <= 0 form: safety_radius^2 - distance^2 <= 0
        # keeps a minimum Euclidean clearance to each modeled obstacle.
        h_terms.append((safety_radius ** 2) - dist_sq)

    model.con_h_expr = ca.vertcat(*h_terms) if h_terms else ca.SX.zeros(0, 1)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = int(horizon)
    ocp.parameter_values = np.zeros((3 * max_obstacles,), dtype=np.float64)

    nx = int(model.x.size()[0])
    nu = int(model.u.size()[0])
    ny = nx + nu

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)
    ocp.cost.yref = np.zeros((ny,), dtype=np.float64)

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref_e = np.zeros((nx,), dtype=np.float64)

    q_pos = 15.0
    q_vel = 1.0
    r_u = 0.2
    # Heuristic weighting rationale:
    # - Position is weighted higher than velocity for goal-reaching priority.
    # - Velocity regularization avoids aggressive terminal overshoot.
    # - Control penalty is moderate so optimizer can still maneuver near threats.
    w = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, r_u, r_u, r_u])
    w_e = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])
    ocp.cost.W = w
    ocp.cost.W_e = w_e

    ocp.constraints.lbu = np.array([-max_accel, -max_accel, -max_accel], dtype=np.float64)
    ocp.constraints.ubu = np.array([max_accel, max_accel, max_accel], dtype=np.float64)
    # Acceleration bounds approximate actuator authority and stabilize optimization.
    ocp.constraints.idxbu = np.array([0, 1, 2], dtype=np.int64)

    ocp.constraints.idxbx = np.array([3, 4, 5], dtype=np.int64)
    # Velocity bounds keep dynamics in the same regime as other baselines.
    ocp.constraints.lbx = np.array([-max_speed, -max_speed, -max_speed], dtype=np.float64)
    ocp.constraints.ubx = np.array([max_speed, max_speed, max_speed], dtype=np.float64)

    nh = max_obstacles
    if nh > 0:
        # Lower bound set to -inf and upper bound to zero enforces h(x,p) <= 0.
        ocp.constraints.lh = np.full((nh,), -ACADOS_INFTY, dtype=np.float64)
        ocp.constraints.uh = np.zeros((nh,), dtype=np.float64)

    ocp.solver_options.tf = float(horizon * dt)
    # Solver choices prioritize real-time repeatability in benchmark loops:
    # - ERK: fast explicit integration for simple point-mass model.
    # - SQP_RTI: single-iteration RTI style for low per-step latency.
    # - GAUSS_NEWTON + HPIPM partial condensing: robust default combination.
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = max(1, horizon // 2)

    return ocp

from __future__ import annotations


def build_point_mass_model(ca):
    model_name = "point_mass_3d"

    px = ca.SX.sym("px")
    py = ca.SX.sym("py")
    pz = ca.SX.sym("pz")
    vx = ca.SX.sym("vx")
    vy = ca.SX.sym("vy")
    vz = ca.SX.sym("vz")
    x = ca.vertcat(px, py, pz, vx, vy, vz)

    ax = ca.SX.sym("ax")
    ay = ca.SX.sym("ay")
    az = ca.SX.sym("az")
    u = ca.vertcat(ax, ay, az)

    xdot = ca.SX.sym("xdot", 6)

    f_expl = ca.vertcat(vx, vy, vz, ax, ay, az)
    f_impl = xdot - f_expl

    return {
        "name": model_name,
        "x": x,
        "u": u,
        "xdot": xdot,
        "f_expl": f_expl,
        "f_impl": f_impl,
    }

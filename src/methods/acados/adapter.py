from __future__ import annotations

from bruce_code.methods.acados.controller import AcadosNMPCController

from ..common import GenericPolicyAdapter


def build_acados_adapter(device_name: str, backend: str = "auto"):
    controller = AcadosNMPCController(backend=backend)
    return GenericPolicyAdapter(
        f"acados_{controller.backend}",
        controller,
        device_name,
        metadata={"backend_requested": backend, "backend_used": controller.backend},
    )

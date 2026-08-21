"""Industrial edge runtime layer (peripheral simulation of an edge PC).

Sits between the camera adapter and the (unchanged) D3 inference service:

    Camera Adapter -> Edge Runtime -> D3 Inference -> Decision Service -> PLC/MES

Provides service lifecycle management, resource monitoring and configuration
driven by ``edge_config.yaml``. It never touches the D3 model artifacts.
"""
from industrial_runtime.config import EdgeConfig, load_edge_config
from industrial_runtime.resource_monitor import ResourceMonitor, RuntimeMetrics
from industrial_runtime.runtime_manager import (
    EdgeRuntimeManager,
    RuntimeState,
    ServiceSpec,
)

__all__ = [
    "EdgeConfig",
    "EdgeRuntimeManager",
    "ResourceMonitor",
    "RuntimeMetrics",
    "RuntimeState",
    "ServiceSpec",
    "load_edge_config",
]

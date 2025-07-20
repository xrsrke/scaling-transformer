"""
Communication Latency and Bandwidth Analysis for Device Mesh Topologies

This module provides tools to calculate communication latency and bandwidth
for various device mesh topologies, particularly focusing on TPU and GPU clusters.
Based on the analysis from "How to Scale Your Model" parts 2 and 3.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import math
import numpy as np


class DeviceType(Enum):
    """Supported device types with their characteristics."""
    TPU_V3 = "tpu_v3"
    TPU_V4 = "tpu_v4" 
    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"
    TPU_V6E = "tpu_v6e"
    GPU_H100 = "gpu_h100"
    GPU_A100 = "gpu_a100"


class TopologyType(Enum):
    """Supported mesh topology types."""
    TORUS_2D = "torus_2d"
    TORUS_3D = "torus_3d"
    RING = "ring"
    TREE = "tree"
    ALL_TO_ALL = "all_to_all"


@dataclass
class HardwareCharacteristics:
    """
    Hardware characteristics for different accelerator types.
    All bandwidth values are in bytes/second.
    All latency values are in seconds.
    """
    # Device identification
    device_type: DeviceType
    model_name: str
    
    # Compute characteristics
    bf16_flops_per_sec: float  # BF16 FLOPs per second per chip
    
    # Memory characteristics
    hbm_capacity_bytes: float  # HBM capacity per chip
    hbm_bandwidth_bytes_per_sec: float  # HBM bandwidth per chip
    
    # Inter-chip communication (ICI)
    ici_bandwidth_unidirectional: float  # Unidirectional ICI bandwidth per link
    ici_bandwidth_bidirectional: float  # Bidirectional ICI bandwidth per link
    
    # Optional compute characteristics
    int8_flops_per_sec: Optional[float] = None  # INT8 ops per second per chip
    ici_latency_per_hop: float = 1e-6  # Per-hop latency in seconds
    ici_num_links: int = 4  # Number of ICI links per device
    
    # Host communication (PCIe)
    pcie_bandwidth_bytes_per_sec: float = 1.5e10  # PCIe bandwidth per chip
    
    # Data center network (DCN)
    dcn_bandwidth_bytes_per_sec: float = 2.5e10  # DCN bandwidth per host
    
    # Topology constraints
    max_pod_size: Tuple[int, ...] = field(default_factory=lambda: (16, 16))
    host_topology: Tuple[int, ...] = field(default_factory=lambda: (4, 2))
    supports_wraparound: bool = True
    wraparound_threshold: int = 16  # Minimum axis size for wraparound


# Predefined hardware configurations based on Part 2 specifications
HARDWARE_CONFIGS = {
    DeviceType.TPU_V3: HardwareCharacteristics(
        device_type=DeviceType.TPU_V3,
        model_name="TPU v3",
        bf16_flops_per_sec=1.4e14,
        int8_flops_per_sec=1.4e14,
        hbm_capacity_bytes=32e9,  # 32GB
        hbm_bandwidth_bytes_per_sec=9.0e11,
        ici_bandwidth_unidirectional=1e11,
        ici_bandwidth_bidirectional=2e11,
        ici_num_links=4,
        max_pod_size=(32, 32),
        host_topology=(4, 2)
    ),
    
    DeviceType.TPU_V4: HardwareCharacteristics(
        device_type=DeviceType.TPU_V4,
        model_name="TPU v4",
        bf16_flops_per_sec=2.75e14,
        int8_flops_per_sec=2.75e14,
        hbm_capacity_bytes=32e9,  # 32GB
        hbm_bandwidth_bytes_per_sec=1.2e12,
        ici_bandwidth_unidirectional=4.5e10,
        ici_bandwidth_bidirectional=9e10,
        ici_num_links=6,  # 3D torus
        max_pod_size=(16, 16, 16),
        host_topology=(2, 2, 1)
    ),
    
    DeviceType.TPU_V5E: HardwareCharacteristics(
        device_type=DeviceType.TPU_V5E,
        model_name="TPU v5e",
        bf16_flops_per_sec=1.97e14,
        int8_flops_per_sec=3.94e14,
        hbm_capacity_bytes=16e9,  # 16GB
        hbm_bandwidth_bytes_per_sec=8.1e11,
        ici_bandwidth_unidirectional=4.5e10,
        ici_bandwidth_bidirectional=9e10,
        ici_num_links=4,  # 2D torus
        max_pod_size=(16, 16),
        host_topology=(4, 2)
    ),
    
    DeviceType.TPU_V5P: HardwareCharacteristics(
        device_type=DeviceType.TPU_V5P,
        model_name="TPU v5p",
        bf16_flops_per_sec=4.59e14,
        int8_flops_per_sec=9.18e14,
        hbm_capacity_bytes=96e9,  # 96GB
        hbm_bandwidth_bytes_per_sec=2.8e12,
        ici_bandwidth_unidirectional=9e10,
        ici_bandwidth_bidirectional=1.8e11,
        ici_num_links=6,  # 3D torus
        max_pod_size=(16, 20, 28),
        host_topology=(2, 2, 1)
    ),
    
    DeviceType.TPU_V6E: HardwareCharacteristics(
        device_type=DeviceType.TPU_V6E,
        model_name="TPU v6e (Trillium)",
        bf16_flops_per_sec=9.20e14,
        int8_flops_per_sec=1.84e15,
        hbm_capacity_bytes=32e9,  # 32GB
        hbm_bandwidth_bytes_per_sec=1.6e12,
        ici_bandwidth_unidirectional=9e10,
        ici_bandwidth_bidirectional=1.8e11,
        ici_num_links=4,  # 2D torus
        max_pod_size=(16, 16),
        host_topology=(4, 2)
    ),
}


@dataclass
class MeshTopology:
    """
    Represents a device mesh topology with communication characteristics.
    """
    shape: Tuple[int, ...]  # Shape of the device mesh (e.g., (8, 8) for 8x8)
    topology_type: TopologyType
    axis_names: Tuple[str, ...] = field(default_factory=lambda: ('X', 'Y', 'Z'))
    
    def __post_init__(self):
        if len(self.shape) != len(self.axis_names[:len(self.shape)]):
            self.axis_names = tuple(f"axis_{i}" for i in range(len(self.shape)))
    
    @property
    def total_devices(self) -> int:
        """Total number of devices in the mesh."""
        return math.prod(self.shape)
    
    @property
    def num_dimensions(self) -> int:
        """Number of dimensions in the mesh."""
        return len(self.shape)
    
    def has_wraparound(self, hardware: HardwareCharacteristics) -> Dict[str, bool]:
        """Check which axes have wraparound connections."""
        wraparound = {}
        for i, (axis_name, axis_size) in enumerate(zip(self.axis_names, self.shape)):
            wraparound[axis_name] = (
                hardware.supports_wraparound and 
                axis_size >= hardware.wraparound_threshold
            )
        return wraparound


class CommunicationAnalyzer:
    """
    Analyzes communication latency and bandwidth for device mesh topologies.
    """
    
    def __init__(self, hardware: HardwareCharacteristics, mesh: MeshTopology):
        self.hardware = hardware
        self.mesh = mesh
        self.wraparound = mesh.has_wraparound(hardware)
    
    def calculate_hop_distance(self, src: Tuple[int, ...], dst: Tuple[int, ...]) -> int:
        """
        Calculate minimum hop distance between two devices in the mesh.
        
        Args:
            src: Source device coordinates
            dst: Destination device coordinates
            
        Returns:
            Minimum number of hops between devices
        """
        if len(src) != len(dst) or len(src) != self.mesh.num_dimensions:
            raise ValueError("Coordinate dimensions must match mesh dimensions")
        
        total_hops = 0
        for i, (s, d, axis_size) in enumerate(zip(src, dst, self.mesh.shape)):
            axis_name = self.mesh.axis_names[i]
            direct_distance = abs(d - s)
            
            if self.wraparound.get(axis_name, False):
                # With wraparound, take the shorter path
                wraparound_distance = axis_size - direct_distance
                hop_distance = min(direct_distance, wraparound_distance)
            else:
                hop_distance = direct_distance
            
            total_hops += hop_distance
        
        return total_hops
    
    def calculate_point_to_point_latency(self, src: Tuple[int, ...], dst: Tuple[int, ...]) -> float:
        """
        Calculate point-to-point communication latency between two devices.
        
        Args:
            src: Source device coordinates
            dst: Destination device coordinates
            
        Returns:
            Latency in seconds
        """
        num_hops = self.calculate_hop_distance(src, dst)
        return num_hops * self.hardware.ici_latency_per_hop
    
    def calculate_allgather_time(self, 
                                data_size_bytes: float, 
                                axis_name: str,
                                multi_axis: bool = False) -> Tuple[float, bool]:
        """
        Calculate AllGather communication time.
        
        Args:
            data_size_bytes: Total size of data to gather
            axis_name: Name of the axis to gather along
            multi_axis: Whether gathering along multiple axes
            
        Returns:
            Tuple of (time_seconds, is_latency_bound)
        """
        try:
            axis_idx = self.mesh.axis_names.index(axis_name)
            axis_size = self.mesh.shape[axis_idx]
        except ValueError:
            raise ValueError(f"Axis {axis_name} not found in mesh")
        
        # Determine effective bandwidth
        num_axes = len([ax for ax in self.mesh.axis_names if ax.startswith(axis_name[0])]) if multi_axis else 1
        effective_bandwidth = self.hardware.ici_bandwidth_bidirectional * num_axes
        
        # Calculate bandwidth-bound time
        bandwidth_time = data_size_bytes / effective_bandwidth
        
        # Calculate latency-bound time
        max_hops = axis_size // 2 if self.wraparound.get(axis_name, False) else axis_size - 1
        latency_time = max_hops * self.hardware.ici_latency_per_hop
        
        # Return the maximum (bottleneck)
        total_time = max(bandwidth_time, latency_time)
        is_latency_bound = latency_time > bandwidth_time
        
        return total_time, is_latency_bound
    
    def calculate_allreduce_time(self, 
                                data_size_bytes: float, 
                                axis_name: str) -> Tuple[float, bool]:
        """
        Calculate AllReduce communication time (2x AllGather).
        
        Args:
            data_size_bytes: Size of data to reduce
            axis_name: Name of the axis to reduce along
            
        Returns:
            Tuple of (time_seconds, is_latency_bound)
        """
        allgather_time, is_latency_bound = self.calculate_allgather_time(data_size_bytes, axis_name)
        return 2 * allgather_time, is_latency_bound
    
    def calculate_reduce_scatter_time(self, 
                                     data_size_bytes: float, 
                                     axis_name: str) -> Tuple[float, bool]:
        """
        Calculate ReduceScatter communication time (same as AllGather).
        
        Args:
            data_size_bytes: Size of data to reduce and scatter
            axis_name: Name of the axis to reduce along
            
        Returns:
            Tuple of (time_seconds, is_latency_bound)
        """
        return self.calculate_allgather_time(data_size_bytes, axis_name)
    
    def calculate_alltoall_time(self, 
                               data_size_bytes: float, 
                               axis_name: str) -> Tuple[float, bool]:
        """
        Calculate AllToAll communication time (1/4 of AllGather for bidirectional ring).
        
        Args:
            data_size_bytes: Size of data for AllToAll
            axis_name: Name of the axis for AllToAll
            
        Returns:
            Tuple of (time_seconds, is_latency_bound)
        """
        allgather_time, is_latency_bound = self.calculate_allgather_time(data_size_bytes, axis_name)
        return allgather_time / 4, is_latency_bound
    
    def analyze_matmul_communication(self, 
                                   matrix_a_shape: Tuple[int, ...],
                                   matrix_b_shape: Tuple[int, ...],
                                   sharding_a: str,
                                   sharding_b: str,
                                   output_sharding: str,
                                   dtype_bytes: int = 2) -> Dict[str, any]:
        """
        Analyze communication requirements for a sharded matrix multiplication.
        
        Args:
            matrix_a_shape: Shape of matrix A
            matrix_b_shape: Shape of matrix B  
            sharding_a: Sharding specification for A (e.g., "I_X,J")
            sharding_b: Sharding specification for B (e.g., "J,K_Y")
            output_sharding: Desired output sharding (e.g., "I_X,K_Y")
            dtype_bytes: Bytes per element (2 for bfloat16, 1 for int8)
            
        Returns:
            Dictionary with communication analysis
        """
        # This is a simplified analysis - in practice would need full sharding parser
        analysis = {
            "operation": f"{sharding_a} @ {sharding_b} -> {output_sharding}",
            "communication_needed": False,
            "communication_type": None,
            "data_size_bytes": 0,
            "communication_time": 0.0,
            "is_latency_bound": False
        }
        
        # Basic heuristics for common cases
        if "_X" in sharding_a and "_X" not in sharding_b and "J" in sharding_a:
            # Case 2: AllGather needed on contracting dimension
            analysis["communication_needed"] = True
            analysis["communication_type"] = "AllGather"
            analysis["data_size_bytes"] = math.prod(matrix_a_shape) * dtype_bytes
            time, latency_bound = self.calculate_allgather_time(analysis["data_size_bytes"], "X")
            analysis["communication_time"] = time
            analysis["is_latency_bound"] = latency_bound
        
        return analysis
    
    def get_topology_summary(self) -> Dict[str, any]:
        """Get a summary of the mesh topology and its characteristics."""
        return {
            "mesh_shape": self.mesh.shape,
            "total_devices": self.mesh.total_devices,
            "topology_type": self.mesh.topology_type.value,
            "axis_names": self.mesh.axis_names,
            "wraparound_axes": self.wraparound,
            "hardware_model": self.hardware.model_name,
            "ici_bandwidth_bidirectional": self.hardware.ici_bandwidth_bidirectional,
            "ici_latency_per_hop": self.hardware.ici_latency_per_hop,
            "max_hop_distance": self._calculate_max_hop_distance()
        }
    
    def _calculate_max_hop_distance(self) -> int:
        """Calculate maximum hop distance in the mesh."""
        max_hops = 0
        for i, axis_size in enumerate(self.mesh.shape):
            axis_name = self.mesh.axis_names[i]
            if self.wraparound.get(axis_name, False):
                max_hops += axis_size // 2
            else:
                max_hops += axis_size - 1
        return max_hops


def create_analyzer(device_type: DeviceType, 
                   mesh_shape: Tuple[int, ...],
                   topology_type: TopologyType = TopologyType.TORUS_2D) -> CommunicationAnalyzer:
    """
    Convenience function to create a CommunicationAnalyzer.
    
    Args:
        device_type: Type of device (TPU, GPU, etc.)
        mesh_shape: Shape of the device mesh
        topology_type: Type of mesh topology
        
    Returns:
        Configured CommunicationAnalyzer
    """
    if device_type not in HARDWARE_CONFIGS:
        raise ValueError(f"Unsupported device type: {device_type}")
    
    hardware = HARDWARE_CONFIGS[device_type]
    mesh = MeshTopology(shape=mesh_shape, topology_type=topology_type)
    return CommunicationAnalyzer(hardware, mesh)


# Example usage and testing functions
def example_tpu_v5e_analysis():
    """Example analysis for TPU v5e configuration."""
    analyzer = create_analyzer(DeviceType.TPU_V5E, (4, 4), TopologyType.TORUS_2D)
    
    print("=== TPU v5e 4x4 Communication Analysis ===")
    print(f"Topology: {analyzer.get_topology_summary()}")
    
    # Example: Point-to-point communication
    latency = analyzer.calculate_point_to_point_latency((0, 0), (3, 3))
    print(f"P2P latency from (0,0) to (3,3): {latency*1e6:.1f} μs")
    
    # Example: AllGather a 34MB array (from Part 3 example)
    data_size = 34e6  # 34MB
    time, is_latency_bound = analyzer.calculate_allgather_time(data_size, 'Y')
    print(f"AllGather 34MB along Y axis: {time*1e6:.0f} μs ({'latency' if is_latency_bound else 'bandwidth'} bound)")
    
    # Example: Small array (latency bound)
    small_data = 32e3  # 32KB
    time, is_latency_bound = analyzer.calculate_allgather_time(small_data, 'Y')
    print(f"AllGather 32KB along Y axis: {time*1e6:.0f} μs ({'latency' if is_latency_bound else 'bandwidth'} bound)")


if __name__ == "__main__":
    example_tpu_v5e_analysis()
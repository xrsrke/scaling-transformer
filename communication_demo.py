"""
Interactive Communication Analysis Demo with Gradio and Plotly

Beautiful visualizations for device mesh communication analysis based on
"How to Scale Your Model" parts 2 and 3.
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
from communication_analysis import (
    CommunicationAnalyzer, create_analyzer, DeviceType, TopologyType,
    HardwareCharacteristics, MeshTopology, HARDWARE_CONFIGS
)


def format_bytes(bytes_val):
    """Format bytes with appropriate units."""
    if bytes_val >= 1e12:
        return f"{bytes_val/1e12:.1f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val/1e9:.1f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val/1e6:.1f} MB"
    elif bytes_val >= 1e3:
        return f"{bytes_val/1e3:.1f} KB"
    return f"{int(bytes_val)} B"


def format_time(time_seconds):
    """Format time with appropriate units."""
    if time_seconds >= 1:
        return f"{time_seconds:.2f} s"
    elif time_seconds >= 1e-3:
        return f"{time_seconds*1e3:.1f} ms"
    elif time_seconds >= 1e-6:
        return f"{time_seconds*1e6:.1f} μs"
    else:
        return f"{time_seconds*1e9:.1f} ns"


def format_bandwidth(bw_bytes_per_sec):
    """Format bandwidth with appropriate units."""
    if bw_bytes_per_sec >= 1e12:
        return f"{bw_bytes_per_sec/1e12:.1f} TB/s"
    elif bw_bytes_per_sec >= 1e9:
        return f"{bw_bytes_per_sec/1e9:.1f} GB/s"
    elif bw_bytes_per_sec >= 1e6:
        return f"{bw_bytes_per_sec/1e6:.1f} MB/s"
    return f"{bw_bytes_per_sec/1e3:.1f} KB/s"


def create_mesh_visualization(analyzer):
    """Create a 3D visualization of the device mesh topology."""
    mesh = analyzer.mesh
    
    if len(mesh.shape) == 2:
        # 2D mesh visualization
        x_coords, y_coords = np.meshgrid(range(mesh.shape[0]), range(mesh.shape[1]))
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        z_coords = np.zeros_like(x_coords)
        
        # Create device nodes
        fig = go.Figure()
        
        # Add device nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=12,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            text=[f"({i},{j})" for i, j in zip(x_coords, y_coords)],
            textposition="middle center",
            name='Devices'
        ))
        
        # Add connections (simplified - showing nearest neighbors)
        edge_x, edge_y, edge_z = [], [], []
        for i in range(mesh.shape[0]):
            for j in range(mesh.shape[1]):
                # Horizontal connections
                if i < mesh.shape[0] - 1:
                    edge_x.extend([i, i+1, None])
                    edge_y.extend([j, j, None])
                    edge_z.extend([0, 0, None])
                # Vertical connections
                if j < mesh.shape[1] - 1:
                    edge_x.extend([i, i, None])
                    edge_y.extend([j, j+1, None])
                    edge_z.extend([0, 0, None])
                
                # Wraparound connections if supported
                wraparound = mesh.has_wraparound(analyzer.hardware)
                if wraparound.get('X', False) and i == mesh.shape[0] - 1:
                    edge_x.extend([i, 0, None])
                    edge_y.extend([j, j, None])
                    edge_z.extend([0, 0, None])
                if wraparound.get('Y', False) and j == mesh.shape[1] - 1:
                    edge_x.extend([i, i, None])
                    edge_y.extend([j, 0, None])
                    edge_z.extend([0, 0, None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='gray', width=4),
            name='ICI Links'
        ))
        
    elif len(mesh.shape) == 3:
        # 3D mesh visualization
        coords = [(i, j, k) for i in range(mesh.shape[0]) 
                           for j in range(mesh.shape[1]) 
                           for k in range(mesh.shape[2])]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        z_coords = [c[2] for c in coords]
        
        fig = go.Figure()
        
        # Add device nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='darkblue')
            ),
            text=[f"({i},{j},{k})" for i, j, k in coords],
            textposition="middle center",
            name='Devices'
        ))
    
    else:
        # 1D mesh (ring)
        x_coords = list(range(mesh.shape[0]))
        y_coords = [0] * mesh.shape[0]
        z_coords = [0] * mesh.shape[0]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(size=15, color='lightblue'),
            text=[f"{i}" for i in x_coords],
            name='Devices'
        ))
    
    fig.update_layout(
        title=f"{analyzer.hardware.model_name} Mesh Topology ({mesh.shape})",
        scene=dict(
            xaxis_title=f"{mesh.axis_names[0]} Axis",
            yaxis_title=f"{mesh.axis_names[1]}" if len(mesh.shape) > 1 else "Y",
            zaxis_title=f"{mesh.axis_names[2]}" if len(mesh.shape) > 2 else "Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=500
    )
    
    return fig


def create_communication_heatmap(analyzer, data_sizes):
    """Create a heatmap showing communication times for different operations and data sizes."""
    operations = ['AllGather', 'AllReduce', 'ReduceScatter', 'AllToAll']
    
    # Calculate times for each operation and data size
    times_matrix = []
    is_latency_bound_matrix = []
    
    for op in operations:
        op_times = []
        op_latency_bound = []
        
        for size in data_sizes:
            if op == 'AllGather':
                time, latency_bound = analyzer.calculate_allgather_time(size, 'X')
            elif op == 'AllReduce':
                time, latency_bound = analyzer.calculate_allreduce_time(size, 'X')
            elif op == 'ReduceScatter':
                time, latency_bound = analyzer.calculate_reduce_scatter_time(size, 'X')
            elif op == 'AllToAll':
                time, latency_bound = analyzer.calculate_alltoall_time(size, 'X')
            
            op_times.append(time * 1e6)  # Convert to microseconds
            op_latency_bound.append(latency_bound)
        
        times_matrix.append(op_times)
        is_latency_bound_matrix.append(op_latency_bound)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=times_matrix,
        x=[format_bytes(size) for size in data_sizes],
        y=operations,
        colorscale='Viridis',
        colorbar=dict(title="Time (μs)"),
        text=[[f"{time:.1f}μs<br>{'Latency' if lb else 'Bandwidth'}" 
               for time, lb in zip(row_times, row_lb)]
              for row_times, row_lb in zip(times_matrix, is_latency_bound_matrix)],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Communication Time by Operation and Data Size",
        xaxis_title="Data Size",
        yaxis_title="Operation Type",
        height=400
    )
    
    return fig


def create_scaling_analysis(analyzer, max_data_size=1e9):
    """Create scaling analysis showing bandwidth vs latency bound regimes."""
    data_sizes = np.logspace(3, math.log10(max_data_size), 50)  # 1KB to max_data_size
    
    # Calculate for different operations
    operations = {
        'AllGather': analyzer.calculate_allgather_time,
        'AllReduce': analyzer.calculate_allreduce_time,
        'ReduceScatter': analyzer.calculate_reduce_scatter_time,
        'AllToAll': analyzer.calculate_alltoall_time
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(operations.keys()),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for idx, (op_name, op_func) in enumerate(operations.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        color = colors[idx]
        
        times = []
        latency_bounds = []
        
        for size in data_sizes:
            time, is_latency_bound = op_func(size, 'X')
            times.append(time * 1e6)  # Convert to microseconds
            latency_bounds.append(is_latency_bound)
        
        # Add time curve
        fig.add_trace(
            go.Scatter(
                x=data_sizes,
                y=times,
                mode='lines',
                name=f'{op_name} Time',
                line=dict(color=color, width=3),
                xaxis=f'x{idx+1}' if idx > 0 else 'x',
                yaxis=f'y{idx+1}' if idx > 0 else 'y'
            ),
            row=row, col=col
        )
        
        # Color regions based on latency vs bandwidth bound
        latency_region_x = []
        latency_region_y = []
        bandwidth_region_x = []
        bandwidth_region_y = []
        
        for i, (size, time, is_lat_bound) in enumerate(zip(data_sizes, times, latency_bounds)):
            if is_lat_bound:
                latency_region_x.append(size)
                latency_region_y.append(time)
            else:
                bandwidth_region_x.append(size)
                bandwidth_region_y.append(time)
        
        # Add latency-bound region
        if latency_region_x:
            fig.add_trace(
                go.Scatter(
                    x=latency_region_x,
                    y=latency_region_y,
                    mode='markers',
                    marker=dict(color='red', size=4, opacity=0.6),
                    name='Latency Bound',
                    showlegend=(idx == 0),
                    xaxis=f'x{idx+1}' if idx > 0 else 'x',
                    yaxis=f'y{idx+1}' if idx > 0 else 'y'
                ),
                row=row, col=col
            )
        
        # Add bandwidth-bound region
        if bandwidth_region_x:
            fig.add_trace(
                go.Scatter(
                    x=bandwidth_region_x,
                    y=bandwidth_region_y,
                    mode='markers',
                    marker=dict(color='blue', size=4, opacity=0.6),
                    name='Bandwidth Bound',
                    showlegend=(idx == 0),
                    xaxis=f'x{idx+1}' if idx > 0 else 'x',
                    yaxis=f'y{idx+1}' if idx > 0 else 'y'
                ),
                row=row, col=col
            )
        
        # Update axes for this subplot
        fig.update_xaxes(type="log", title_text="Data Size (bytes)", row=row, col=col)
        fig.update_yaxes(type="log", title_text="Time (μs)", row=row, col=col)
    
    fig.update_layout(
        title="Communication Scaling Analysis: Latency vs Bandwidth Bound",
        height=800,
        showlegend=True
    )
    
    return fig


def analyze_communication(device_type, mesh_x, mesh_y, mesh_z, data_size_mb):
    """Main analysis function called by Gradio interface."""
    # Convert inputs
    device_enum = DeviceType(device_type.lower().replace(" ", "_").replace(".", ""))
    
    # Create mesh shape based on dimensions
    if mesh_z > 1:
        mesh_shape = (mesh_x, mesh_y, mesh_z)
        topology = TopologyType.TORUS_3D
    elif mesh_y > 1:
        mesh_shape = (mesh_x, mesh_y)
        topology = TopologyType.TORUS_2D
    else:
        mesh_shape = (mesh_x,)
        topology = TopologyType.RING
    
    # Create analyzer
    analyzer = create_analyzer(device_enum, mesh_shape, topology)
    
    # Convert data size to bytes
    data_size_bytes = data_size_mb * 1e6
    
    # Get hardware summary
    hardware_summary = f"""
    ## 🔧 Hardware Configuration: {analyzer.hardware.model_name}
    
    **Compute Performance:**
    - BF16 FLOPs/sec: {analyzer.hardware.bf16_flops_per_sec/1e12:.1f} TFLOP/s per chip
    - INT8 FLOPs/sec: {analyzer.hardware.int8_flops_per_sec/1e12:.1f} TFLOP/s per chip
    
    **Memory:**
    - HBM Capacity: {format_bytes(analyzer.hardware.hbm_capacity_bytes)} per chip
    - HBM Bandwidth: {format_bandwidth(analyzer.hardware.hbm_bandwidth_bytes_per_sec)} per chip
    
    **Communication:**
    - ICI Bandwidth: {format_bandwidth(analyzer.hardware.ici_bandwidth_bidirectional)} (bidirectional)
    - ICI Latency: {format_time(analyzer.hardware.ici_latency_per_hop)} per hop
    - PCIe Bandwidth: {format_bandwidth(analyzer.hardware.pcie_bandwidth_bytes_per_sec)}
    - DCN Bandwidth: {format_bandwidth(analyzer.hardware.dcn_bandwidth_bytes_per_sec)}
    """
    
    # Calculate communication times
    operations_data = []
    
    for op_name, op_func in [
        ('AllGather', analyzer.calculate_allgather_time),
        ('AllReduce', analyzer.calculate_allreduce_time),
        ('ReduceScatter', analyzer.calculate_reduce_scatter_time),
        ('AllToAll', analyzer.calculate_alltoall_time)
    ]:
        time, is_latency_bound = op_func(data_size_bytes, 'X')
        operations_data.append({
            'Operation': op_name,
            'Time': format_time(time),
            'Regime': 'Latency Bound' if is_latency_bound else 'Bandwidth Bound',
            'Time (μs)': time * 1e6
        })
    
    # Create operations DataFrame
    operations_df = pd.DataFrame(operations_data)
    
    # Create visualizations
    mesh_fig = create_mesh_visualization(analyzer)
    
    # Data sizes for heatmap (from 1KB to 1GB)
    heatmap_sizes = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    heatmap_fig = create_communication_heatmap(analyzer, heatmap_sizes)
    
    scaling_fig = create_scaling_analysis(analyzer)
    
    # Point-to-point analysis
    if len(mesh_shape) >= 2:
        p2p_latency = analyzer.calculate_point_to_point_latency((0, 0), (mesh_shape[0]-1, mesh_shape[1]-1))
        p2p_analysis = f"""
        ## 📡 Point-to-Point Communication
        
        **Example:** Device (0,0) to Device ({mesh_shape[0]-1},{mesh_shape[1]-1})
        - Hop Distance: {analyzer.calculate_hop_distance((0, 0), (mesh_shape[0]-1, mesh_shape[1]-1))} hops
        - Latency: {format_time(p2p_latency)}
        """
    else:
        p2p_analysis = ""
    
    # Topology summary
    topology_summary = f"""
    ## 🔗 Mesh Topology: {mesh_shape}
    
    **Configuration:**
    - Total Devices: {analyzer.mesh.total_devices:,}
    - Dimensions: {len(mesh_shape)}D {topology.value.replace('_', ' ').title()}
    - Wraparound Support: {list(analyzer.wraparound.keys())}
    
    {p2p_analysis}
    
    ## 📊 Communication Analysis for {format_bytes(data_size_bytes)}
    """
    
    return (
        hardware_summary,
        topology_summary,
        operations_df,
        mesh_fig,
        heatmap_fig,
        scaling_fig
    )


# Create Gradio interface
with gr.Blocks(title="TPU Communication Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Device Mesh Communication Analysis
    
    Interactive analysis tool for TPU/GPU communication patterns based on 
    "How to Scale Your Model" Parts 2 & 3. Explore latency, bandwidth, and 
    communication primitives across different device topologies.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Configuration")
            
            device_type = gr.Dropdown(
                label="Device Type",
                choices=["TPU v3", "TPU v4", "TPU v5e", "TPU v5p", "TPU v6e"],
                value="TPU v5e",
                info="Select the accelerator type"
            )
            
            gr.Markdown("### 📐 Mesh Topology")
            mesh_x = gr.Slider(
                label="X Dimension",
                minimum=1, maximum=32, step=1, value=4,
                info="Number of devices along X axis"
            )
            mesh_y = gr.Slider(
                label="Y Dimension", 
                minimum=1, maximum=32, step=1, value=4,
                info="Number of devices along Y axis"
            )
            mesh_z = gr.Slider(
                label="Z Dimension",
                minimum=1, maximum=16, step=1, value=1,
                info="Number of devices along Z axis (1 for 2D)"
            )
            
            data_size_mb = gr.Slider(
                label="Data Size (MB)",
                minimum=0.001, maximum=1000, step=0.1, value=34,
                info="Size of data for communication analysis"
            )
            
            analyze_btn = gr.Button("🧮 Analyze Communication", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            hardware_info = gr.Markdown(label="Hardware Information")
            topology_info = gr.Markdown(label="Topology Information")
    
    with gr.Tabs():
        with gr.TabItem("📊 Operations Summary"):
            operations_table = gr.DataFrame(
                label="Communication Operations Analysis",
                headers=["Operation", "Time", "Regime", "Time (μs)"]
            )
        
        with gr.TabItem("🔗 Mesh Visualization"):
            mesh_plot = gr.Plot(label="Device Mesh Topology")
        
        with gr.TabItem("🔥 Communication Heatmap"):
            heatmap_plot = gr.Plot(label="Communication Time Heatmap")
        
        with gr.TabItem("📈 Scaling Analysis"):
            scaling_plot = gr.Plot(label="Latency vs Bandwidth Scaling")
    
    # Examples
    gr.Examples(
        examples=[
            ["TPU v5e", 4, 4, 1, 34],      # TPU v5e 4x4, 34MB (from Part 3)
            ["TPU v5p", 4, 4, 4, 100],     # TPU v5p 3D mesh
            ["TPU v6e", 8, 8, 1, 500],     # Large 2D mesh
            ["TPU v4", 2, 2, 2, 10],       # Small 3D mesh, latency bound
        ],
        inputs=[device_type, mesh_x, mesh_y, mesh_z, data_size_mb],
        label="Example Configurations"
    )
    
    # Wire up the analysis
    analyze_btn.click(
        fn=analyze_communication,
        inputs=[device_type, mesh_x, mesh_y, mesh_z, data_size_mb],
        outputs=[hardware_info, topology_info, operations_table, mesh_plot, heatmap_plot, scaling_plot]
    )
    
    # Auto-analyze on load
    demo.load(
        fn=analyze_communication,
        inputs=[device_type, mesh_x, mesh_y, mesh_z, data_size_mb],
        outputs=[hardware_info, topology_info, operations_table, mesh_plot, heatmap_plot, scaling_plot]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
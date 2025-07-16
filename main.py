import gradio as gr
import pandas as pd
import sympy as sp
import plotly.graph_objects as go

def calculate_transformer_stats(batch_size, seq_length, d_model, d_ff, n_heads, n_kv_heads, d_qkv, n_layers, vocab_size):
    """
    Calculate training FLOPs and parameters for a transformer model.
    Based on formulas from "How to Scale Your Model" document.
    """
    B = int(batch_size)
    T = int(seq_length)
    D = int(d_model)
    F = int(d_ff)
    N = int(n_heads)
    K = int(n_kv_heads)
    H = int(d_qkv)
    L = int(n_layers)
    V = int(vocab_size)
    
    # Create symbolic variables
    B_sym, T_sym, D_sym, F_sym, N_sym, K_sym, H_sym, L_sym, V_sym = sp.symbols('B T D F N K H L V')
    
    # Format large numbers
    def format_num(num):
        if num >= 1e15:
            return f"{num/1e15:.2f}P"
        elif num >= 1e12:
            return f"{num/1e12:.2f}T"
        elif num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(int(num))

    # Helper function to convert sympy expression to LaTeX
    def to_latex(expr):
        return f"${sp.latex(expr)}$"

    # MLP FLOPs formulas
    mlp_in1_flops = 6 * B_sym * T_sym * D_sym * F_sym
    mlp_in2_flops = 6 * B_sym * T_sym * D_sym * F_sym
    mlp_act_flops = B_sym * T_sym * F_sym  # O(BTF)
    mlp_out_flops = 6 * B_sym * T_sym * F_sym * D_sym
    mlp_total_flops = mlp_in1_flops + mlp_in2_flops + mlp_out_flops  # Ignoring small activation cost

    # MLP Parameters formulas
    mlp_in1_params = D_sym * F_sym
    mlp_in2_params = D_sym * F_sym
    mlp_out_params = F_sym * D_sym
    mlp_total_params = mlp_in1_params + mlp_in2_params + mlp_out_params

    # Create MLP operations table with LaTeX formulas
    mlp_data = {
        'Operation Name': [
            'Input Projection 1 (Gating)',
            'Input Projection 2 (Value)',
            'Gated Activation',
            'Output Projection',
            '**Total MLP**'
        ],
        'Operation': [
            r'$A[B, T, D] \cdot W_{\text{in1}}[D, F]$',
            r'$A[B, T, D] \cdot W_{\text{in2}}[D, F]$',
            r'$\sigma(A_{\text{in1}})[B, T, F] \odot A_{\text{in2}}[B, T, F]$',
            r'$A[B, T, F] \cdot W_{\text{out}}[F, D]$',
            '**Sum of all MLP operations**'
        ],
        'Train FLOPs Formula': [
            to_latex(mlp_in1_flops),
            to_latex(mlp_in2_flops),
            r'$O(B \cdot T \cdot F)$',
            to_latex(mlp_out_flops),
            to_latex(mlp_total_flops)
        ],
        'Parameters Formula': [
            to_latex(mlp_in1_params),
            to_latex(mlp_in2_params),
            r'$0$',
            to_latex(mlp_out_params),
            to_latex(mlp_total_params)
        ]
    }
    mlp_df = pd.DataFrame(mlp_data)

    # Attention FLOPs formulas
    attn_q_flops = 6 * B_sym * T_sym * D_sym * N_sym * H_sym
    attn_k_flops = 6 * B_sym * T_sym * D_sym * K_sym * H_sym
    attn_v_flops = 6 * B_sym * T_sym * D_sym * K_sym * H_sym
    attn_qk_flops = 6 * B_sym * T_sym**2 * N_sym * H_sym
    attn_softmax_flops = B_sym * T_sym**2 * N_sym  # O(BT²N)
    attn_sv_flops = 6 * B_sym * T_sym**2 * N_sym * H_sym
    attn_o_flops = 6 * B_sym * T_sym * N_sym * H_sym * D_sym
    attn_total_flops = 12 * B_sym * T_sym * D_sym * (N_sym + K_sym) * H_sym + 12 * B_sym * T_sym**2 * N_sym * H_sym

    # Attention Parameters formulas
    attn_q_params = D_sym * N_sym * H_sym
    attn_k_params = D_sym * K_sym * H_sym
    attn_v_params = D_sym * K_sym * H_sym
    attn_o_params = N_sym * H_sym * D_sym
    attn_total_params = 2 * D_sym * (N_sym + K_sym) * H_sym

    # Create Attention operations table with LaTeX formulas
    attention_data = {
        'Operation Name': [
            'Query Projection',
            'Key Projection',
            'Value Projection',
            'QK Dot Product',
            'Softmax',
            'Attention Weights × Values',
            'Output Projection',
            '**Total Attention**'
        ],
        'Operation': [
            r'$A[B, T, D] \cdot W_Q[D, N, H]$',
            r'$A[B, T, D] \cdot W_K[D, K, H]$',
            r'$A[B, T, D] \cdot W_V[D, K, H]$',
            r'$Q[B, T, N, H] \cdot K^T[B, T, K, H]$',
            r'$\text{softmax}_T(QK^T/\sqrt{H})[B, T, T, N]$',
            r'$\text{Attn}[B, T, T, N] \cdot V[B, T, K, H]$',
            r'$A[B, T, N, H] \cdot W_O[N, H, D]$',
            '**Sum of all attention operations**'
        ],
        'Train FLOPs Formula': [
            to_latex(attn_q_flops),
            to_latex(attn_k_flops),
            to_latex(attn_v_flops),
            to_latex(attn_qk_flops),
            r'$O(B \cdot T^2 \cdot N)$',
            to_latex(attn_sv_flops),
            to_latex(attn_o_flops),
            to_latex(attn_total_flops)
        ],
        'Parameters Formula': [
            to_latex(attn_q_params),
            to_latex(attn_k_params),
            to_latex(attn_v_params),
            r'$0$',
            r'$0$',
            r'$0$',
            to_latex(attn_o_params),
            to_latex(attn_total_params)
        ]
    }
    attention_df = pd.DataFrame(attention_data)

    # Transformer block formulas
    layernorm_flops = 2 * B_sym * T_sym * D_sym  # 2 layernorms per block
    layernorm_params = 2 * D_sym
    vocab_flops = 12 * B_sym * T_sym * D_sym * V_sym
    vocab_params = 2 * D_sym * V_sym
    
    # Per-layer totals
    per_layer_flops = mlp_total_flops + attn_total_flops + layernorm_flops
    per_layer_params = mlp_total_params + attn_total_params + layernorm_params
    
    # Total model
    total_model_flops = L_sym * per_layer_flops + vocab_flops
    total_model_params = L_sym * per_layer_params + vocab_params

    # Calculate actual values for analysis and pie chart
    mlp_in1_flops_val = 6 * B * T * D * F
    mlp_in2_flops_val = 6 * B * T * D * F
    mlp_out_flops_val = 6 * B * T * F * D
    mlp_total_flops_val = 18 * B * T * D * F
    mlp_total_params_val = 3 * D * F
    
    attn_q_flops_val = 6 * B * T * D * N * H
    attn_k_flops_val = 6 * B * T * D * K * H
    attn_v_flops_val = 6 * B * T * D * K * H
    attn_qk_flops_val = 6 * B * T * T * N * H
    attn_sv_flops_val = 6 * B * T * T * N * H
    attn_o_flops_val = 6 * B * T * N * H * D
    attention_total_flops_val = 12 * B * T * D * (N + K) * H + 12 * B * T * T * N * H
    attention_total_params_val = 2 * D * (N + K) * H
    
    layernorm_params_val = D
    layernorm_flops_val = 2 * B * T * D  # per layer
    vocab_params_val = 2 * D * V
    vocab_flops_val = 12 * B * T * D * V
    
    total_params_val = L * (mlp_total_params_val + attention_total_params_val + 2 * layernorm_params_val) + vocab_params_val
    total_flops_val = L * (mlp_total_flops_val + attention_total_flops_val + 2 * B * T * D) + 12 * B * T * D * V

    # Create pie chart data
    operations = []
    flops_values = []
    
    # MLP operations (multiply by L for all layers)
    operations.extend([
        'MLP Input Proj 1',
        'MLP Input Proj 2',
        'MLP Output Proj'
    ])
    flops_values.extend([
        mlp_in1_flops_val * L,
        mlp_in2_flops_val * L,
        mlp_out_flops_val * L
    ])
    
    # Attention operations (multiply by L for all layers)
    operations.extend([
        'Attention Q Proj',
        'Attention K Proj',
        'Attention V Proj',
        'Attention QK Product',
        'Attention Weights×Values',
        'Attention Output Proj'
    ])
    flops_values.extend([
        attn_q_flops_val * L,
        attn_k_flops_val * L,
        attn_v_flops_val * L,
        attn_qk_flops_val * L,
        attn_sv_flops_val * L,
        attn_o_flops_val * L
    ])
    
    # Other operations
    operations.extend([
        'LayerNorm',
        'Vocabulary (Embed/Unembed)'
    ])
    flops_values.extend([
        layernorm_flops_val * L,
        vocab_flops_val
    ])
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=operations,
        values=flops_values,
        textinfo='label+percent',
        textposition='auto',
        hole=0.3,
        marker=dict(
            colors=['#FF6B6B', '#FD7E14', '#FFA94D', '#69DB7C', '#74C0FC', '#4DABF7', 
                   '#748FFC', '#9775FA', '#DA77F2', '#F06595', '#FFD43B']
        )
    )])
    
    fig.update_layout(
        title={
            'text': f'Training FLOPs Distribution<br><sub>Total: {format_num(sum(flops_values))} FLOPs</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        font=dict(size=12)
    )

    transformer_data = {
        'Component': [
            'MLP Block',
            'Attention Block',
            'LayerNorm (×2)',
            '**Total per Layer**',
            '',
            'Vocabulary (embed/unembed)',
            f'**Total Model (L={L} layers)**'
        ],
        'Train FLOPs Formula': [
            to_latex(mlp_total_flops),
            to_latex(attn_total_flops),
            to_latex(layernorm_flops),
            to_latex(per_layer_flops),
            '',
            to_latex(vocab_flops),
            to_latex(total_model_flops)
        ],
        'Parameters Formula': [
            to_latex(mlp_total_params),
            to_latex(attn_total_params),
            to_latex(layernorm_params),
            to_latex(per_layer_params),
            '',
            to_latex(vocab_params),
            to_latex(total_model_params)
        ]
    }
    transformer_df = pd.DataFrame(transformer_data)

    # Analysis text
    analysis = f"""
### 📊 Model Analysis

**Architecture Summary:**
- Model Size: {format_num(total_params_val)} parameters ({total_params_val/1e9:.1f}B)
- Training FLOPs: {format_num(total_flops_val)}
- FLOPs per Token: {format_num(total_flops_val / (B * T))}

**Component Breakdown:**
- MLP dominates with {(mlp_total_params_val * L / total_params_val * 100):.1f}% of parameters
- Attention uses {(attention_total_params_val * L / total_params_val * 100):.1f}% of parameters
- Vocabulary uses {(vocab_params_val / total_params_val * 100):.1f}% of parameters

**Performance Insights:**
- Attention becomes FLOPs-dominant when T > {8 * D:,} tokens
- Current sequence length: {T:,} tokens
- Status: {"Attention is dominant" if T > 8 * D else "MLP is dominant"}
- 6×params×tokens approximation: {total_flops_val / (6 * total_params_val * B * T):.2f}×
"""

    return mlp_df, attention_df, transformer_df, analysis, fig

# Create Gradio interface with tabs
with gr.Blocks(title="Transformer Math Calculator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧮 Transformer Math: Complete Breakdown
    
    This calculator shows the detailed mathematical breakdown of transformer operations, 
    following the formulas from "How to Scale Your Model". Each operation is shown 
    individually before being combined into totals.
    """)
    
    with gr.Tabs():
        with gr.TabItem("🔧 Configuration"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Training Configuration")
                    batch_size = gr.Number(label="Batch Size (B)", value=1024, info="Number of sequences")
                    seq_length = gr.Number(label="Sequence Length (T)", value=2048, info="Tokens per sequence")
                    
                    gr.Markdown("### Model Architecture")
                    d_model = gr.Number(label="Model Dimension (D)", value=4096, info="Hidden dimension")
                    d_ff = gr.Number(label="FFN Dimension (F)", value=16384, info="Feed-forward dimension")
                    n_layers = gr.Number(label="Number of Layers (L)", value=32, info="Transformer layers")
                    vocab_size = gr.Number(label="Vocabulary Size (V)", value=50000, info="Token vocabulary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Attention Configuration")
                    n_heads = gr.Number(label="Number of Heads (N)", value=32, info="Query/Output heads")
                    n_kv_heads = gr.Number(label="KV Heads (K)", value=32, info="Key-value heads (for GQA/MQA)")
                    d_qkv = gr.Number(label="Head Dimension (H)", value=128, info="Dimension per attention head")
                    
                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=[
                            [1024, 2048, 4096, 16384, 32, 32, 128, 32, 50000],  # ~7B model
                            [4096, 4096, 8192, 28672, 64, 8, 128, 80, 128256],  # LLaMA-3 70B style
                            [2048, 8192, 12288, 49152, 96, 96, 128, 96, 50257],  # ~175B GPT-3 style
                            [512, 512, 768, 3072, 12, 12, 64, 12, 30522],       # BERT-base style
                        ],
                        inputs=[batch_size, seq_length, d_model, d_ff, n_heads, n_kv_heads, d_qkv, n_layers, vocab_size],
                        label="Common Model Configurations"
                    )
            
            calculate_btn = gr.Button("🧮 Calculate Transformer Math", variant="primary", size="lg")
        
        with gr.TabItem("📊 Transformer Math"):
            gr.Markdown("## MLP Operations Breakdown")
            gr.Markdown("The MLP block uses gating with three matrices: W_in1, W_in2 (gated), and W_out:")
            mlp_table = gr.DataFrame(
                label="MLP Operations (per layer)",
                headers=["Operation Name", "Operation", "Train FLOPs Formula", "Parameters Formula"],
                datatype=["str", "str", "str", "str"],
                row_count=(5, "fixed")
            )
            
            gr.Markdown("## Attention Operations Breakdown")
            gr.Markdown("Attention includes Q, K, V projections, dot-product attention, and output projection:")
            attention_table = gr.DataFrame(
                label="Attention Operations (per layer)",
                headers=["Operation Name", "Operation", "Train FLOPs Formula", "Parameters Formula"],
                datatype=["str", "str", "str", "str"],
                row_count=(8, "fixed")
            )
            
            gr.Markdown("## Complete Transformer Block")
            gr.Markdown("Summary combining all components:")
            transformer_table = gr.DataFrame(
                label="Transformer Block Summary",
                headers=["Component", "Train FLOPs Formula", "Parameters Formula"],
                datatype=["str", "str", "str"],
                row_count=(7, "fixed")
            )
            
            analysis_output = gr.Markdown(label="Analysis")
            
        with gr.TabItem("📈 FLOPs Distribution"):
            gr.Markdown("## FLOPs Distribution by Operation")
            gr.Markdown("This pie chart shows the percentage of training FLOPs consumed by each operation across all layers.")
            flops_chart = gr.Plot(label="FLOPs Distribution")
    
    # Wire up the calculation
    calculate_btn.click(
        fn=calculate_transformer_stats,
        inputs=[batch_size, seq_length, d_model, d_ff, n_heads, n_kv_heads, d_qkv, n_layers, vocab_size],
        outputs=[mlp_table, attention_table, transformer_table, analysis_output, flops_chart]
    )
    
    # Auto-calculate on load
    demo.load(
        fn=calculate_transformer_stats,
        inputs=[batch_size, seq_length, d_model, d_ff, n_heads, n_kv_heads, d_qkv, n_layers, vocab_size],
        outputs=[mlp_table, attention_table, transformer_table, analysis_output, flops_chart]
    )

if __name__ == "__main__":
    demo.launch()
    
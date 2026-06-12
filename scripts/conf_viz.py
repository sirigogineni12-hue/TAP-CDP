import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import os

# Set style for a clean academic look
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

def create_architecture_overview():
    """Creates the end-to-end pipeline diagram using Graphviz."""
    dot = graphviz.Digraph(comment='TAP-HybridNet Architecture', format='png')
    dot.attr(rankdir='LR', size='10,5')
    
    # Define nodes
    dot.node('A', 'Input Image\n(RGB)', shape='box', style='filled', fillcolor='lightgray')
    dot.node('B', 'JPEG Block DCT\n(8x8 Blocks)', shape='parallelogram')
    dot.node('C', 'DCT Coefficients\n(64 Channels)', shape='folder')
    
    with dot.subgraph(name='cluster_frontend') as c:
        c.attr(label='TAP Frontend', style='dashed', color='blue')
        c.node('D', 'TACS Module\n(Learnable Gating)', shape='component', color='blue')
        c.node('E', 'Sigmoid & \nThreshold (M)', shape='diamond', color='blue')
    
    dot.node('F', 'Sparse DCT Volume\n(N < 64 channels)', shape='folder', fillcolor='lightblue', style='filled')
    
    with dot.subgraph(name='cluster_backend') as c2:
        c2.attr(label='Hybrid Backbone', style='dashed', color='darkgreen')
        c2.node('G', 'Depthwise Separable\nConvolutions', shape='box')
        c2.node('H', 'Multi-Head\nSelf-Attention', shape='box')
    
    dot.node('I', 'Semantic Labels\n(Classification)', shape='ellipse', style='filled', fillcolor='lightgreen')

    # Connect nodes
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')

    dot.render('architecture_overview', cleanup=True)

def create_hybrid_block():
    """Creates the internal Hybrid block diagram."""
    dot = graphviz.Digraph(comment='Hybrid Block', format='png')
    dot.attr(rankdir='TB')
    
    dot.node('In', 'Input Feature Map', shape='none')
    dot.node('DW', 'Depthwise Conv (3x3)', shape='box')
    dot.node('BN1', 'BatchNorm & Leaky ReLU', shape='box')
    dot.node('PW', 'Pointwise Conv (1x1)', shape='box')
    dot.node('Attn', 'Multi-Head Self-Attention', shape='box', color='red')
    dot.node('Add', '+', shape='circle')
    dot.node('Out', 'Output Feature Map', shape='none')
    
    dot.edge('In', 'DW')
    dot.edge('DW', 'BN1')
    dot.edge('BN1', 'PW')
    dot.edge('PW', 'Attn')
    dot.edge('Attn', 'Add')
    dot.edge('In', 'Add', label='Residual Link', constraint='false', style='dotted')
    dot.edge('Add', 'Out')
    
    dot.render('hybrid_block', cleanup=True)

def plot_latency():
    """Generates the latency comparison chart."""
    data = {
        'Model': ['MobileNetV2\n(Pixel)', 'ResNet-18\n(Pixel)', 'HybridNet\n(Pixel)', 'Naive DCT\nHybrid', 'TAP-HybridNet\n(Ours)'],
        'Preprocessing': [8.4, 8.4, 8.4, 1.1, 1.1],
        'Inference': [6.4, 24.1, 6.8, 6.2, 4.1]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    p1 = plt.barh(df['Model'], df['Preprocessing'], color='lightcoral', label='Preprocessing (IDCT/Extraction)')
    p2 = plt.barh(df['Model'], df['Inference'], left=df['Preprocessing'], color='skyblue', label='Inference (Forward Pass)')
    
    plt.xlabel('Latency (ms)')
    plt.title('End-to-End Latency Breakdown on NVIDIA T4 (CIFAR-10)')
    plt.legend(loc='lower right')
    plt.gca().invert_yaxis()
    
    for i, (p, inf) in enumerate(zip(df['Preprocessing'], df['Inference'])):
        plt.text(p + inf + 0.5, i, f'{p+inf:.1f}ms', va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)

def plot_pareto():
    """Generates the Accuracy vs Sparsity Pareto frontier."""
    # Mock data based on paper results
    active_coeffs = np.array([64, 52, 40, 32, 26, 18, 12, 8])
    accuracy = np.array([69.1, 69.0, 68.9, 68.8, 68.55, 67.2, 65.1, 61.5])
    
    plt.figure(figsize=(8, 6))
    plt.plot(active_coeffs, accuracy, marker='o', linestyle='--', color='navy', linewidth=2, label='TAP-HybridNet')
    
    # Annotate key point
    plt.annotate('Ours (60% reduction)', 
                 xy=(26, 68.55), xytext=(35, 65),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
    
    # Baseline line
    plt.axhline(y=69.0, color='gray', linestyle=':', label='Pixel Baseline (HybridNet)')
    
    plt.xlabel('Active DCT Coefficients (Channels)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Pareto Frontier: Accuracy vs. Input Complexity')
    plt.xlim(70, 0) # Invert X to show increasing pruning
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pareto_frontier.png', dpi=300)

if __name__ == "__main__":
    create_architecture_overview()
    create_hybrid_block()
    plot_latency()
    plot_pareto()
    print("All figures generated successfully.")
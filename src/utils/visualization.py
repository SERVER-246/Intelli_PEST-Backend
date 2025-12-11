import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(55, 100))
ax.set_xlim(0, 10)
ax.set_ylim(0, 50)
ax.axis('off')

# Define colors
colors = {
    'start_end': '#87CEEB',
    'init': '#B0C4DE',
    'backbone': '#DDA0DD',
    'ensemble': '#98FB98',
    'super': '#FFB6C1',
    'onnx': '#FFD700',
    'decision': '#FFA07A'
}

def draw_box(ax, x, y, width, height, text, color, style='round'):
    """Draw a rounded rectangle box"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=f"round,pad=0.1" if style == 'round' else "round,pad=0.05",
                         linewidth=2, edgecolor='black', facecolor=color,
                         alpha=0.8, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, 
            weight='bold', zorder=3, wrap=True)

def draw_oval(ax, x, y, width, height, text, color):
    """Draw an oval shape"""
    ellipse = mpatches.Ellipse((x, y), width, height, 
                               linewidth=2, edgecolor='black', 
                               facecolor=color, alpha=0.8, zorder=2)
    ax.add_patch(ellipse)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=10, weight='bold', zorder=3)

def draw_diamond(ax, x, y, width, height, text, color):
    """Draw a diamond shape for decision points"""
    points = [
        [x, y + height/2],  # top
        [x + width/2, y],   # right
        [x, y - height/2],  # bottom
        [x - width/2, y]    # left
    ]
    diamond = mpatches.Polygon(points, closed=True, linewidth=2,
                              edgecolor='black', facecolor=color,
                              alpha=0.8, zorder=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=8, weight='bold', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black', zorder=1)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                style='italic', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='none'))

# Y positions
y_pos = 48
y_step = 2.5

# Draw Pipeline
# Start
draw_oval(ax, 5, y_pos, 2, 1, 'Pipeline Start', colors['start_end'])
y_pos -= y_step

# Initialize
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3, 1, 'Initialize Environment', colors['init'])
y_pos -= y_step

# Data Prep
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Prepare Dataset & DataLoaders', colors['init'])
y_pos -= y_step

# Backbone Training Section
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Train Individual Backbones', colors['backbone'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3, 1, 'K-Fold Cross Validation', colors['backbone'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 2.8, 1, 'Train Classifier Head', colors['backbone'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 2.8, 1, 'Fine-Tune Full Model', colors['backbone'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.2, 1, 'Evaluate & Export Backbone', colors['backbone'])
y_pos -= y_step

# Decision: All Backbones Trained?
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.6)
draw_diamond(ax, 5, y_pos, 3, 1.2, 'All Backbones\nTrained?', colors['decision'])

# Loop back arrow
draw_arrow(ax, 5 + 1.5, y_pos, 8, y_pos, 'No')
draw_arrow(ax, 8, y_pos, 8, y_pos + 6 * y_step)
draw_arrow(ax, 8, y_pos + 6 * y_step, 5 + 1.75, y_pos + 6 * y_step)

y_pos -= y_step

# Continue to Ensemble
draw_arrow(ax, 5, y_pos + y_step - 0.6, 5, y_pos + 0.5, 'Yes')

# Ensemble Training Section
draw_box(ax, 5, y_pos, 3, 1, 'Train Ensemble Models', colors['ensemble'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Create Ensemble Architecture', colors['ensemble'])
y_pos -= y_step

# Fusion Type Decision
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.6)
draw_diamond(ax, 5, y_pos, 2.2, 1.2, 'Fusion Type', colors['decision'])

# Three fusion branches
fusion_y = y_pos - y_step
# Attention
draw_arrow(ax, 5 - 1.1, y_pos - 0.4, 2.5, fusion_y + 0.5)
draw_box(ax, 2.5, fusion_y, 2.5, 1, 'Attention Weights\n+ MLP', colors['ensemble'])
draw_arrow(ax, 2.5, fusion_y - 0.5, 2.5, fusion_y - y_step + 0.5)

# Cross
draw_arrow(ax, 5, y_pos - 0.6, 5, fusion_y + 0.5)
draw_box(ax, 5, fusion_y, 2.5, 1, 'Cross-Attention\n+ MLP', colors['ensemble'])
draw_arrow(ax, 5, fusion_y - 0.5, 5, fusion_y - y_step + 0.5)

# Concat
draw_arrow(ax, 5 + 1.1, y_pos - 0.4, 7.5, fusion_y + 0.5)
draw_box(ax, 7.5, fusion_y, 2.2, 1, 'Deep Fusion\nMLP', colors['ensemble'])
draw_arrow(ax, 7.5, fusion_y - 0.5, 7.5, fusion_y - y_step + 0.5)

y_pos = fusion_y - y_step

# Converge branches
draw_arrow(ax, 2.5, y_pos + 0.5, 5, y_pos + 0.5)
draw_arrow(ax, 7.5, y_pos + 0.5, 5, y_pos + 0.5)

draw_box(ax, 5, y_pos, 2.8, 1, 'Train Fusion Layers', colors['ensemble'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.2, 1, 'Evaluate & Export Ensemble', colors['ensemble'])
y_pos -= y_step

# Decision: All Fusion Types Done?
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.6)
draw_diamond(ax, 5, y_pos, 3, 1.2, 'All Fusion\nTypes Done?', colors['decision'])

# Loop back arrow for ensembles
draw_arrow(ax, 5 + 1.5, y_pos, 8.5, y_pos, 'No')
draw_arrow(ax, 8.5, y_pos, 8.5, y_pos + 5 * y_step)
draw_arrow(ax, 8.5, y_pos + 5 * y_step, 5 + 1.5, y_pos + 5 * y_step)

y_pos -= y_step

# Continue to Super Ensemble
draw_arrow(ax, 5, y_pos + y_step - 0.6, 5, y_pos + 0.5, 'Yes')

# Super Ensemble Section
draw_box(ax, 5, y_pos, 3, 1, 'Train Super Ensemble', colors['super'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Load 3 Trained Ensembles', colors['super'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Create Meta-Fusion Network', colors['super'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3, 1, 'Train Meta-Fusion Only', colors['super'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.8, 1, 'Evaluate & Export Super Ensemble', colors['super'])
y_pos -= y_step

# ONNX Conversion Section
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 2.5, 1, 'ONNX Conversion', colors['onnx'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Convert All Models to ONNX', colors['onnx'])
y_pos -= y_step

draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_box(ax, 5, y_pos, 3.5, 1, 'Create Deployment Packages', colors['onnx'])
y_pos -= y_step

# End
draw_arrow(ax, 5, y_pos + y_step - 0.5, 5, y_pos + 0.5)
draw_oval(ax, 5, y_pos, 2.5, 1, 'Pipeline Complete', colors['start_end'])

# Add title
plt.title('Pest Classification Pipeline - Complete Flow', 
          fontsize=16, weight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['start_end'], edgecolor='black', label='Start/End'),
    mpatches.Patch(facecolor=colors['init'], edgecolor='black', label='Initialization'),
    mpatches.Patch(facecolor=colors['backbone'], edgecolor='black', label='Backbone Training'),
    mpatches.Patch(facecolor=colors['ensemble'], edgecolor='black', label='Ensemble Training'),
    mpatches.Patch(facecolor=colors['super'], edgecolor='black', label='Super Ensemble'),
    mpatches.Patch(facecolor=colors['onnx'], edgecolor='black', label='ONNX Export'),
    mpatches.Patch(facecolor=colors['decision'], edgecolor='black', label='Decision Point')
]
ax.legend(handles=legend_elements, loc='upper right', 
          fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('pest_classification_pipeline.png', dpi=300, bbox_inches='tight')
print("Pipeline flowchart saved as 'pest_classification_pipeline.png'")
plt.show()
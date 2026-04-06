
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Define positions for nodes (x, y coordinates)
# Layer 1: USER INPUT
start_y = 5
input_y = 4.2
preprocess_y = 3
model_y = 2
embedding_y = 1.3
fusion_y = 0.5
output_y = -0.5

# Define colors for each stage
input_color = '#ADD8E6'  # Light blue
preprocess_color = '#90EE90'  # Light green
model_color = '#FFD580'  # Light orange
fusion_color = '#E6D5FF'  # Light purple
output_color = '#FFB6C1'  # Light red

# Helper function to add a box
def add_box(fig, x, y, text, color, width=0.8, height=0.3):
    fig.add_shape(
        type="rect",
        x0=x-width/2, y0=y-height/2,
        x1=x+width/2, y1=y+height/2,
        line=dict(color="#333333", width=2),
        fillcolor=color
    )
    fig.add_annotation(
        x=x, y=y,
        text=text,
        showarrow=False,
        font=dict(size=10, color="#13343b"),
        align="center"
    )

# Helper function to add an arrow
def add_arrow(fig, x0, y0, x1, y1):
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#333333"
    )

# Layer 1: Start
add_box(fig, 0, start_y, "USER INPUT", input_color, width=1.0)

# Layer 2: Three parallel inputs
add_box(fig, -1.5, input_y, "Behavioral Data<br>CSV/JSON", input_color, width=0.9)
add_box(fig, 0, input_y, "Voice Recording<br>Audio", input_color, width=0.9)
add_box(fig, 1.5, input_y, "Facial Image<br>Photo", input_color, width=0.9)

# Layer 3: Preprocessing
add_box(fig, -1.5, preprocess_y, "Feature Scaling<br>Normalization", preprocess_color, width=0.9)
add_box(fig, 0, preprocess_y, "MFCC Extract<br>librosa", preprocess_color, width=0.9)
add_box(fig, 1.5, preprocess_y, "Image Preproc<br>Resizing", preprocess_color, width=0.9)

# Layer 4: Models
add_box(fig, -1.5, model_y, "XGBoost<br>Classifier", model_color, width=0.9)
add_box(fig, 0, model_y, "CNN-LSTM<br>Network", model_color, width=0.9)
add_box(fig, 1.5, model_y, "ResNet50<br>Transfer Learn", model_color, width=0.9)

# Layer 5: Embeddings
add_box(fig, -1.5, embedding_y, "2D Embedding", model_color, width=0.9, height=0.25)
add_box(fig, 0, embedding_y, "32D Embedding", model_color, width=0.9, height=0.25)
add_box(fig, 1.5, embedding_y, "64D Embedding", model_color, width=0.9, height=0.25)

# Layer 6: Fusion
add_box(fig, 0, fusion_y, "Concatenate<br>Embeddings 98D", fusion_color, width=1.2)
add_box(fig, 0, fusion_y-0.5, "Dense Fusion<br>Network", fusion_color, width=1.2)

# Layer 7: Outputs
add_box(fig, -1.5, output_y, "Final Predict", output_color, width=0.7, height=0.25)
add_box(fig, -0.5, output_y, "Risk Assess<br>L/M/H", output_color, width=0.7, height=0.25)
add_box(fig, 0.5, output_y, "Confidence<br>Score", output_color, width=0.7, height=0.25)
add_box(fig, 1.5, output_y, "Individual<br>Results", output_color, width=0.7, height=0.25)

# Add arrows
# Start to inputs
add_arrow(fig, 0, start_y-0.2, -1.5, input_y+0.15)
add_arrow(fig, 0, start_y-0.2, 0, input_y+0.15)
add_arrow(fig, 0, start_y-0.2, 1.5, input_y+0.15)

# Inputs to preprocessing
add_arrow(fig, -1.5, input_y-0.15, -1.5, preprocess_y+0.15)
add_arrow(fig, 0, input_y-0.15, 0, preprocess_y+0.15)
add_arrow(fig, 1.5, input_y-0.15, 1.5, preprocess_y+0.15)

# Preprocessing to models
add_arrow(fig, -1.5, preprocess_y-0.15, -1.5, model_y+0.15)
add_arrow(fig, 0, preprocess_y-0.15, 0, model_y+0.15)
add_arrow(fig, 1.5, preprocess_y-0.15, 1.5, model_y+0.15)

# Models to embeddings
add_arrow(fig, -1.5, model_y-0.15, -1.5, embedding_y+0.125)
add_arrow(fig, 0, model_y-0.15, 0, embedding_y+0.125)
add_arrow(fig, 1.5, model_y-0.15, 1.5, embedding_y+0.125)

# Embeddings to fusion
add_arrow(fig, -1.5, embedding_y-0.125, -0.5, fusion_y+0.15)
add_arrow(fig, 0, embedding_y-0.125, 0, fusion_y+0.15)
add_arrow(fig, 1.5, embedding_y-0.125, 0.5, fusion_y+0.15)

# Fusion concatenate to fusion network
add_arrow(fig, 0, fusion_y-0.15, 0, fusion_y-0.35)

# Fusion to outputs
add_arrow(fig, 0, fusion_y-0.65, -1.5, output_y+0.125)
add_arrow(fig, 0, fusion_y-0.65, -0.5, output_y+0.125)
add_arrow(fig, 0, fusion_y-0.65, 0.5, output_y+0.125)
add_arrow(fig, 0, fusion_y-0.65, 1.5, output_y+0.125)

# Update layout
fig.update_layout(
    title="Multimodal ASD Screening System",
    xaxis=dict(visible=False, range=[-2.5, 2.5]),
    yaxis=dict(visible=False, range=[-1.2, 5.5]),
    plot_bgcolor='#F3F3EE',
    paper_bgcolor='#F3F3EE',
    showlegend=False
)

# Save the figure
fig.write_image('flowchart.png')
fig.write_image('flowchart.svg', format='svg')

print("Flowchart saved successfully!")

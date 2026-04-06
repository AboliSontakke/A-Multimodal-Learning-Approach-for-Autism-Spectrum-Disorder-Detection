
import plotly.graph_objects as go
import pandas as pd

# Data
data = [
    {"Model": "Behavioral", "Accuracy": 87, "Precision": 85, "Recall": 83, "F1-Score": 84, "AUROC": 87},
    {"Model": "Voice", "Accuracy": 84, "Precision": 82, "Recall": 80, "F1-Score": 81, "AUROC": 84},
    {"Model": "Facial", "Accuracy": 90, "Precision": 88, "Recall": 86, "F1-Score": 87, "AUROC": 89},
    {"Model": "Fusion", "Accuracy": 93, "Precision": 91, "Recall": 89, "F1-Score": 90, "AUROC": 92}
]

df = pd.DataFrame(data)

# Brand colors for the 5 metrics
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C']

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC']

# Create figure
fig = go.Figure()

# Add bars for each metric
for i, metric in enumerate(metrics):
    fig.add_trace(go.Bar(
        name=metric,
        x=df['Model'],
        y=df[metric],
        marker_color=colors[i],
        text=df[metric],
        textposition='outside',
        texttemplate='%{y}'
    ))

# Update layout
fig.update_layout(
    title='Expected Model Performance Comparison',
    xaxis_title='Model',
    yaxis_title='Performance (%)',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
    yaxis=dict(range=[0, 100])
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('chart.png')
fig.write_image('chart.svg', format='svg')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Step 1: Mock Data Simulation ---
# Structure is based on the paper's definition: Sample = {question, reference_answer, model_answer, metrics, judge_explanations} 
mock_data = {
    "question": "What are the key benefits of using RAG systems in healthcare applications?",
    "reference_answer": "Key benefits include: 1) Enhanced accuracy in medical research and guidelines, 2) Reduction of outdated information through grounded responses, and 3) Better compliance with regulatory standards.",
    "model_answer": "RAG systems help doctors by providing quick answers from a vast knowledge base, which improves diagnostic accuracy and supports continuous medical education.",
    "metrics": {
        "Factual Accuracy": 0.90,
        "Coherence": 0.95,
        "Relevance": 0.88,
        "Completeness": 0.75,
        "Conciseness": 0.80
    },
    "judge_explanations": {
        "Factual Accuracy": "The model correctly identifies diagnostic accuracy as a benefit, which aligns with the reference's point about enhanced accuracy.",
        "Coherence": "The response is well-structured and easy to follow.",
        "Relevance": "The answer is highly relevant to the healthcare context of the query.",
        "Completeness": "The answer is missing key points mentioned in the reference, such as regulatory compliance and reduction of outdated information.",
        "Conciseness": "The answer is direct and to the point."
    }
}

# --- Step 2: Build the Streamlit Interface ---

# Page configuration
st.set_page_config(layout="wide", page_title="ADAMS Interactive Evaluation")

# Title, referencing the paper's ADAMS name
st.title("ADAMS: Interactive Evaluation Interface")
st.markdown("An interactive visualization interface designed to bring transparency to the ADAMS framework.")

# --- Sidebar: Metric Weight Controls ---
# This corresponds to the "Metric Control Matrix" and "Dynamic Weight Adjustment" mentioned in the paper.
st.sidebar.header("Metric Control Matrix")
st.sidebar.markdown("Adjust the weights of each metric in real-time to see its impact on the final score.")

# Create a dictionary to store the weights adjusted by the user
weights = {}
for metric_name in mock_data["metrics"].keys():
    # Use sliders for user weight adjustment
    weights[metric_name] = st.sidebar.slider(
        label=metric_name,
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Default weight
        step=0.05
    )

# --- Main Page: Display Results ---

# Calculate the final weighted score, based on the formula S_final = Σ(w_i * M_i) from the paper 
metric_scores = mock_data["metrics"]
final_score = sum(metric_scores[metric] * weights[metric] for metric in metric_scores)
total_weight = sum(weights.values())

# Avoid division by zero
normalized_score = final_score / total_weight if total_weight > 0 else 0

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    # Display the final score, referencing the "Final ADAMS Score" box in Figure 3 of the paper 
    st.subheader("Final ADAMS Score")
    st.metric(label="Normalized Weighted Score", value=f"{normalized_score:.2f}")
    
    # Use Plotly to create a bar chart for visualization, as mentioned in the paper's "Metric Visualization" section 
    st.subheader("Metric Scores Breakdown")
    metric_names = list(metric_scores.keys())
    scores = list(metric_scores.values())
    
    fig = go.Figure([go.Bar(x=metric_names, y=scores, text=scores, textposition='auto')])
    fig.update_layout(title_text="Per-Metric Scores", xaxis_title="Metrics", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)


with col2:
    # Display the content of the sample itself
    st.subheader("Sample Analysis")
    st.text_area("Question", mock_data["question"], height=100)
    st.text_area("Reference Answer", mock_data["reference_answer"], height=150)
    st.text_area("Model Answer", mock_data["model_answer"], height=150)
    
    # Display judge explanations using an expander, as mentioned in the paper's "Judge Explanation Viewer" 
    st.subheader("Judge Explanations")
    for metric_name, explanation in mock_data["judge_explanations"].items():
        with st.expander(f"**{metric_name}**: {metric_scores[metric_name]:.2f}"):
            st.write(explanation)
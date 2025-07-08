import streamlit as st
import time
import json
import pandas as pd
import random
from datetime import datetime
from backend_client import get_backend_client
import os
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="ADAMS - RAG Evaluation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for both light and dark themes
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Detect system theme preference */
    :root {
        --primary-bg: #ffffff;
        --secondary-bg: #f8f9fa;
        --card-bg: #ffffff;
        --text-primary: #1a1a1a;
        --text-secondary: #6c757d;
        --accent-color: #2563eb;
        --accent-light: rgba(37, 99, 235, 0.1);
        --border-color: rgba(0, 0, 0, 0.1);
        --shadow-color: rgba(0, 0, 0, 0.1);
        --success-color: #22c55e;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-bg: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
            --secondary-bg: rgba(15, 15, 25, 0.8);
            --card-bg: rgba(15, 15, 25, 0.9);
            --text-primary: #ffffff;
            --text-secondary: #b8bcc8;
            --accent-color: #00f5ff;
            --accent-light: rgba(0, 245, 255, 0.1);
            --border-color: rgba(0, 245, 255, 0.3);
            --shadow-color: rgba(0, 245, 255, 0.1);
            --success-color: #06ffa5;
            --warning-color: #ffa500;
            --error-color: #ff006e;
        }
    }
    
    /* Main app styling */
    .stApp {
        background: var(--primary-bg);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, var(--error-color), #8338ec, var(--accent-color), var(--success-color));
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Card styling - adaptive */
    .cyber-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .cyber-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px var(--shadow-color);
    }
    
    /* Text styling */
    .neon-text {
        color: var(--accent-color);
        text-shadow: 0 0 10px var(--accent-color);
        font-weight: 600;
    }
    
    /* Metric cards - improved visibility */
    .metric-display {
        background: var(--card-bg);
        border: 2px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px var(--shadow-color);
    }
    
    .metric-display:hover {
        border-color: var(--accent-color);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px var(--shadow-color);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-color);
        text-shadow: 0 0 20px var(--accent-light);
    }
    
    .metric-name {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Score display - enhanced visibility */
    .score-display {
        text-align: center;
        padding: 2rem;
        background: var(--accent-light);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 2px solid var(--accent-color);
        box-shadow: 0 8px 32px var(--shadow-color);
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-color);
        text-shadow: 0 0 30px var(--accent-light);
    }
    
    /* Improved sidebar styling */
    .css-1d391kg {
        background: var(--secondary-bg);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-color);
        color: var(--primary-bg);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--accent-light);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--accent-light);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-radius: 8px;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--card-bg);
        border: 2px dashed var(--border-color);
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-color);
        background: var(--accent-light);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: var(--accent-color);
    }
    
    /* Alert styling */
    .stAlert {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    /* Success alert */
    .stAlert[data-baseweb="notification"] [data-testid="alertSuccessContent"] {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid var(--success-color);
    }
    
    /* Warning alert */
    .stAlert[data-baseweb="notification"] [data-testid="alertWarningContent"] {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid var(--warning-color);
    }
    
    /* Error alert */
    .stAlert[data-baseweb="notification"] [data-testid="alertErrorContent"] {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--error-color);
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        color: var(--text-primary);
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-color);
        color: var(--primary-bg);
        border-color: var(--accent-color);
    }
    
    /* Metric styling */
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Custom status indicators */
    .status-online {
        color: var(--success-color);
        font-weight: 600;
    }
    
    .status-offline {
        color: var(--error-color);
        font-weight: 600;
    }
    
    /* Comment section */
    .comment-section {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px var(--shadow-color);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'data_annotation'
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = 'OpenAI GPT-4o-mini (Recommended)'
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'comparison_selection' not in st.session_state:
    st.session_state.comparison_selection = {'eval_a': None, 'eval_b': None}
if 'original_dataset' not in st.session_state:
    st.session_state.original_dataset = None
if 'error_dataset' not in st.session_state:
    st.session_state.error_dataset = None
if 'backend_client' not in st.session_state:
    st.session_state.backend_client = get_backend_client()

# Sample metrics data
default_metrics = {
    'Factual Accuracy': {'score': 8.7, 'weight': 0.9},
    'Coherence': {'score': 9.2, 'weight': 0.8},
    'Relevance': {'score': 8.9, 'weight': 0.85},
    'Completeness': {'score': 7.8, 'weight': 0.7},
    'Citation Quality': {'score': 8.1, 'weight': 0.75},
    'Domain Specificity': {'score': 8.5, 'weight': 0.8},
    'Clarity': {'score': 9.0, 'weight': 0.7},
    'Consistency': {'score': 8.3, 'weight': 0.6},
    'Novelty': {'score': 7.5, 'weight': 0.5},
    'Readability': {'score': 8.8, 'weight': 0.6},
    'Technical Depth': {'score': 8.0, 'weight': 0.7},
    'Evidence Support': {'score': 8.4, 'weight': 0.8},
    'Contextual Fit': {'score': 8.6, 'weight': 0.7},
    'Timeliness': {'score': 7.9, 'weight': 0.6},
    'Bias Detection': {'score': 8.2, 'weight': 0.7}
}

def find_score_column_for_evaluator(df_columns, evaluator_name):
    """Find the appropriate score column for an evaluator in the dataframe with enhanced matching"""
    # Clean evaluator name for matching
    clean_evaluator = evaluator_name.replace('Evaluator', '').lower().replace('_', ' ')
    base_name = clean_evaluator.replace(' ', '').lower()
    
    # Debug: Print what we're looking for
    print(f"Looking for columns for evaluator: {evaluator_name}")
    print(f"Base name: {base_name}")
    print(f"Available columns: {list(df_columns)}")
    
    # Extended patterns with more specific matching
    patterns = [
        # Most common pattern based on the debug output
        f"gold_FACTUAL_ACCURACY_accuracy_score" if "factualaccuracy" in base_name else None,
        f"gold_factual_correctness_F1_score" if "factualcorrectness" in base_name else None,
        f"gold_key_point_completeness_score" if "keypointcompleteness" in base_name else None,
        f"gold_key_point_hallucination_score" if "keypointhallucinati" in base_name else None,
        f"gold_Context_Relevance_relevance_score" if "contextrelevance" in base_name else None,
        f"gold_COHERENCE_coherence_score" if "coherence" in base_name else None,
        f"gold_engagement_engagement_score" if "engagement" in base_name else None,
        
        # Generic patterns
        f"gold_{base_name}_score",
        f"gold_{clean_evaluator.replace(' ', '_')}_score",
        f"{evaluator_name.lower()}_score",
        f"{clean_evaluator.replace(' ', '_')}_score", 
        f"{clean_evaluator.replace(' ', '')}_score",
        
        # Without 'score' suffix
        f"gold_{base_name}",
        f"gold_{clean_evaluator.replace(' ', '_')}",
        f"{evaluator_name.lower()}",
        clean_evaluator.replace(' ', '_'),
        clean_evaluator.replace(' ', ''),
        
        # Fallback patterns
        f"{base_name}_evaluation",
        f"{clean_evaluator}_evaluation",
        f"{base_name}",
    ]
    
    # Remove None patterns
    patterns = [p for p in patterns if p is not None]
    
    # Try exact matches first
    for pattern in patterns:
        for col in df_columns:
            if col.lower() == pattern.lower():
                print(f"Exact match found: {col} for pattern {pattern}")
                return col
    
    # Try case-insensitive contains matches
    for pattern in patterns:
        for col in df_columns:
            if pattern.lower() in col.lower():
                print(f"Contains match found: {col} for pattern {pattern}")
                return col
    
    # Try reverse matching - look for evaluator parts in column names
    evaluator_keywords = [
        "factual", "accuracy", "correctness", "completeness", 
        "hallucination", "context", "relevance", "coherence", "engagement"
    ]
    
    relevant_keywords = [kw for kw in evaluator_keywords if kw in base_name]
    
    for col in df_columns:
        col_lower = col.lower()
        if 'score' in col_lower:
            # Check if evaluator keywords appear in column
            keyword_matches = sum(1 for kw in relevant_keywords if kw in col_lower)
            if keyword_matches >= len(relevant_keywords) * 0.5:  # 50% keyword match
                print(f"Keyword match found: {col} (keywords: {relevant_keywords})")
                return col
    
    print(f"No match found for {evaluator_name}")
    return None

def recalculate_final_score(evaluation_result, updated_weights):
    """Recalculate the final evaluation score based on updated weights"""
    try:
        processed_dataset = evaluation_result.get("processed_dataset", [])
        if not processed_dataset:
            return None
            
        df = pd.DataFrame(processed_dataset)
        df["Final_Score"] = 0.0
        
        total_weight = 0.0
        # Calculate weighted sum
        for evaluator_name, weight in updated_weights.items():
            score_col = find_score_column_for_evaluator(df.columns, evaluator_name)
            if score_col:
                df["Final_Score"] += df[score_col] * weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            df["Final_Score"] = df["Final_Score"] / total_weight if total_weight != 1.0 else df["Final_Score"]
        
        # Calculate new final average score
        new_final_score = df["Final_Score"].mean()
        
        # Update evaluation_result
        updated_result = evaluation_result.copy()
        updated_result["final_score"] = new_final_score
        updated_result["processed_dataset"] = df.to_dict('records')
        
        return updated_result
        
    except Exception as e:
        st.error(f"❌ Error recalculating scores: {str(e)}")
        return None

def display_metric_scores_with_adjustment(evaluation_result, selected_metrics):
    """Display metric scores with improved weight adjustment using number inputs"""
    if not evaluation_result or "processed_dataset" not in evaluation_result:
        st.warning("⚠️ No evaluation data to display")
        return
    
    try:
        processed_dataset = evaluation_result["processed_dataset"]
        if not processed_dataset:
            st.warning("⚠️ Processed dataset is empty")
            return
        
        df = pd.DataFrame(processed_dataset)
        
        # Display final score prominently at the very top
        final_score = evaluation_result.get("final_score", 0)
        st.markdown("""
        <div style="text-align: left; margin: 20px 0;">
                                    <h2 style="color: var(--text-secondary); font-size: 1.5rem; margin: 0;">Final Evaluation Score</h2>
                                    <h1 style="color: var(--accent-color); font-size: 4rem; margin: 10px 0; font-weight: bold; text-shadow: 0 0 20px var(--accent-light);">
                    🎯 {:.3f}
                </h1>
        </div>
        """.format(final_score), unsafe_allow_html=True)
        
        st.subheader("📊 Metric Scores & Weight Adjustment")
        
        # Create two-column layout
        col_scores, col_weights = st.columns([1, 1])
        
        with col_scores:
            st.markdown("### 📈 Individual Metric Scores")
            
            # Display statistics for each metric's scores
            metric_stats = {}
            
            for evaluator_name in selected_metrics.keys():
                score_col = find_score_column_for_evaluator(df.columns, evaluator_name)
                
                if score_col:
                    scores = df[score_col]
                    # Handle NaN values
                    valid_scores = scores.dropna()
                    if len(valid_scores) > 0:
                        metric_stats[evaluator_name] = {
                            "mean": valid_scores.mean(),
                            "std": valid_scores.std(),
                            "min": valid_scores.min(),
                            "max": valid_scores.max(),
                            "column": score_col,
                            "valid_count": len(valid_scores),
                            "sample_scores": valid_scores.tolist()[:3]
                        }
                    else:
                        st.warning(f"⚠️ No valid scores found for {evaluator_name}")
                else:
                    st.error(f"❌ Score column not found for {evaluator_name}")
                    # Show all available columns for debugging
                    st.write("**Available columns:**", list(df.columns))
            
            # Display metric statistics
            if metric_stats:
                for evaluator_name, stats in metric_stats.items():
                    with st.expander(f"📊 {evaluator_name.replace('Evaluator', '')}", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Average", f"{stats['mean']:.3f}")
                        with col2:
                            st.metric("Std Dev", f"{stats['std']:.3f}")
                        with col3:
                            st.metric("Min", f"{stats['min']:.3f}")
                        with col4:
                            st.metric("Max", f"{stats['max']:.3f}")
                        
                        # Display additional info
                        st.write(f"**Column Used:** `{stats['column']}`")
                        st.write(f"**Valid Samples:** {stats['valid_count']}")
                        st.write(f"**Sample Scores:** {stats['sample_scores']}")
            else:
                st.error("❌ No matching score columns found")
        
        with col_weights:
            st.markdown("### ⚖️ Weight Adjustment")
            
            # Use sliders for intuitive weight adjustment
            st.info("💡 Drag sliders to adjust weights. Changes are applied in real-time.")
            
            # Initialize stable weight storage in session state
            weights_key = f"weights_{hash(str(sorted(selected_metrics.keys())))}"
            if weights_key not in st.session_state:
                st.session_state[weights_key] = selected_metrics.copy()
            
            # Track weight changes to prevent unnecessary recalculations
            if f"{weights_key}_last_total" not in st.session_state:
                st.session_state[f"{weights_key}_last_total"] = sum(selected_metrics.values())
            
            updated_weights = {}
            total_weight = 0.0
            
            # Create number input for each metric with stable keys
            for i, (evaluator_name, original_weight) in enumerate(selected_metrics.items()):
                # Clean display name
                display_name = evaluator_name.replace('Evaluator', '').replace('_', ' ')
                
                # Use stable weight value from session state
                current_weight = st.session_state[weights_key].get(evaluator_name, original_weight)
                
                col_name, col_input = st.columns([2, 1])
                
                with col_name:
                    st.markdown(f"**{display_name}**")
                
                with col_input:
                    # Use extremely stable key that never changes during session
                    input_key = f"stable_weight_{evaluator_name}_{weights_key}"
                    new_weight = st.slider(
                        "Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_weight),
                        step=0.01,
                        key=input_key,
                        label_visibility="collapsed"
                    )
                
                # Update weights tracking
                st.session_state[weights_key][evaluator_name] = new_weight
                updated_weights[evaluator_name] = new_weight
                total_weight += new_weight
                
                # Show weight as percentage
                st.progress(new_weight, text=f"{new_weight:.1%}")
            
            # Display total weight with color coding
            weight_color = "green" if abs(total_weight - 1.0) < 0.05 else "orange"
            st.markdown(f"**Total Weight: <span style='color: {weight_color}'>{total_weight:.3f}</span>**", 
                       unsafe_allow_html=True)
            
            # Auto-update final score when weights change (with throttling)
            last_total = st.session_state.get(f"{weights_key}_last_total", sum(selected_metrics.values()))
            weights_changed = abs(total_weight - last_total) > 0.01
            
            if weights_changed and total_weight > 0:
                # Update the last total to prevent repeated calculations
                st.session_state[f"{weights_key}_last_total"] = total_weight
                normalized_weights = {k: v/total_weight for k, v in updated_weights.items()}
                
                # Auto-calculate and display updated score
                updated_result = recalculate_final_score(evaluation_result, normalized_weights)
                if updated_result:
                    original_score = evaluation_result.get("final_score", 0)
                    new_score = updated_result["final_score"]
                    delta = new_score - original_score
                    
                    st.markdown("---")
                    st.markdown("### 🔄 Real-time Score Update")
                    
                    # Show updated score prominently
                    delta_color = "var(--success-color)" if delta >= 0 else "var(--error-color)"
                    st.markdown("""
                    <div style="text-align: center; margin: 10px 0; padding: 15px; background: var(--accent-light); border: 1px solid var(--accent-color); border-radius: 10px;">
                        <h3 style="color: var(--accent-color); font-size: 2rem; margin: 0; text-shadow: 0 0 15px var(--accent-light);">
                            {:.3f}
                        </h3>
                        <p style="color: {}; font-size: 1rem; margin: 5px 0; font-weight: 600;">
                            {} {:.3f} from original
                        </p>
                    </div>
                    """.format(
                        new_score,
                        delta_color,
                        "+" if delta >= 0 else "",
                        delta
                    ), unsafe_allow_html=True)
                    
                    # Store updated result in session state
                    st.session_state.updated_evaluation_result = updated_result
                    st.session_state.current_weights = normalized_weights
                    
                    # Also update the current evaluation result for future displays
                    st.session_state.current_evaluation_result = updated_result
            
            # Action buttons with unique keys and session state handling
            col_norm, col_reset = st.columns(2)
            
            with col_norm:
                normalize_key = f"normalize_btn_{weights_key}"
                if st.button("🔧 Normalize to 100%", help="Automatically adjust weights to sum to 1.0", key=normalize_key):
                    if total_weight > 0:
                        # Normalize weights and update session state using stable keys
                        for evaluator_name in selected_metrics.keys():
                            normalized_value = updated_weights[evaluator_name] / total_weight
                            st.session_state[weights_key][evaluator_name] = normalized_value
                        # Reset last total to trigger recalculation
                        st.session_state[f"{weights_key}_last_total"] = 1.0
                        st.success("✅ Weights normalized to 100%")
            
            with col_reset:
                reset_key = f"reset_btn_{weights_key}"
                if st.button("🔄 Reset Weights", help="Reset all weights to original values", key=reset_key):
                    # Reset to original weights using stable storage
                    for evaluator_name, original_weight in selected_metrics.items():
                        st.session_state[weights_key][evaluator_name] = float(original_weight)
                    # Reset last total to trigger recalculation
                    st.session_state[f"{weights_key}_last_total"] = sum(selected_metrics.values())
                    st.success("✅ Weights reset to original values")
                    
    except Exception as e:
        st.error(f"❌ Error displaying metric scores: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

def display_discussion_summary_with_sliders(discussion_summary, selected_metrics):
    """Display discussion summary with only rationale text"""
    
    with st.expander("🤖 AI Agent Discussion Summary - including rationale", expanded=False):
        
        # Parse and display only rationale content
        if discussion_summary and isinstance(discussion_summary, str):
            try:
                # Try to parse JSON if it's JSON format
                import json
                if discussion_summary.strip().startswith('{'):
                    parsed_data = json.loads(discussion_summary)
                    
                    # Extract and display only rationale text
                    rationale = parsed_data.get("rationale", "")
                    if rationale:
                        st.info(rationale)
                    else:
                        st.info("No rationale available from agent discussion.")
                else:
                    # Display as plain text
                    st.info(discussion_summary)
                    
            except json.JSONDecodeError:
                # Fallback to plain text display
                st.info(discussion_summary)
        else:
            st.info("No detailed rationale available from agent discussion.")


def clean_dataset(df):
    """Clean dataset by removing NaN values and empty rows, and map columns according to constants.py"""
    # Remove rows where all values are NaN
    df_cleaned = df.dropna(how='all')
    
    # Column mapping based on constants.py RAGBENCH_COL_NAMES
    required_columns = ['question', 'response', 'documents']
    
    # Try to map existing columns to expected ones
    column_mapping = {}
    for col in df_cleaned.columns:
        col_lower = col.lower().strip()
        if col_lower in ['question', 'query', 'q']:
            column_mapping[col] = 'question'
        elif col_lower in ['response', 'answer', 'reference_answer', 'golden_answer', 'ground_truth']:
            column_mapping[col] = 'response'
        elif col_lower in ['documents', 'context', 'doc', 'document', 'contexts']:
            column_mapping[col] = 'documents'
    
    # Apply column mapping
    if column_mapping:
        df_cleaned = df_cleaned.rename(columns=column_mapping)
    
    # Handle key_points column specifically
    key_points_mapping = {}
    for col in df_cleaned.columns:
        col_lower = col.lower().strip()
        if col_lower in ['key_points', 'keypoints', 'key-points', 'key points']:
            key_points_mapping[col] = 'key_points'
    
    if key_points_mapping:
        df_cleaned = df_cleaned.rename(columns=key_points_mapping)
    
    # Check if we have all required columns
    existing_columns = [col for col in required_columns if col in df_cleaned.columns]
    
    if len(existing_columns) < 3:
        # Try to use first 3 columns as fallback
        cols = df_cleaned.columns.tolist()
        if len(cols) >= 3:
            # Create new dataframe with mapped columns but preserve other columns
            new_df = df_cleaned.copy()
            for i, req_col in enumerate(required_columns):
                if i < len(cols):
                    new_df[req_col] = df_cleaned.iloc[:, i]
                    if cols[i] != req_col and cols[i] in new_df.columns:
                        new_df = new_df.drop(columns=[cols[i]])
            df_cleaned = new_df
        else:
            return None, "Dataset must have at least 3 columns: question, response, documents"
    # If we have all required columns, keep the dataframe as is (preserving additional columns like key_points)
    
    # Remove rows with NaN in required columns
    df_cleaned = df_cleaned.dropna(subset=required_columns)
    
    # Replace any remaining NaN values with empty strings
    df_cleaned = df_cleaned.fillna('')
    
    return df_cleaned, f"Cleaned dataset: {len(df_cleaned)} valid samples"


def run_original_annotation_pipeline(df):
    """Run the original annotation pipeline using ExecutionPipeline"""
    try:
        # Clean dataset first and ensure correct column mapping
        df_cleaned, message = clean_dataset(df)
        if df_cleaned is None:
            return None, message
        
        # Convert to format compatible with ExecutionPipeline
        dataset = df_cleaned.to_dict('records')
        
        # Use backend API to run the original pipeline - fix the result structure
        result = st.session_state.backend_client.start_data_annotation(
            dataset=dataset,
            llm_provider="openai",
            model_name="gpt-4o-mini-2024-07-18",
            base_url="https://api.openai.com/v1/",
            selected_error_types=["Entity_Error", "Negation", "Missing_Information", "Out_of_Reference", "Numerical_Error"],
            error_probabilities=[0.0, 0.7, 0.3],
            include_key_points=True
        )
        
        # Backend returns 'annotated_dataset' not 'augmented_dataset'
        if result and 'annotated_dataset' in result:
            annotated_df = pd.DataFrame(result['annotated_dataset'])
            return annotated_df, f"Successfully processed {len(annotated_df)} samples"
        else:
            return None, f"Annotation pipeline failed: {result}"
            
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg and "Invalid API key" in error_msg:
            return None, "OpenAI API key is invalid or not set. Please check your OPENAI_API_KEY environment variable."
        elif "500 Server Error" in error_msg and "annotation" in error_msg:
            return None, "Backend annotation service failed. This might be due to network connectivity issues or missing OpenAI API key."
        else:
            return None, f"Pipeline execution failed: {error_msg}"


def generate_synthetic_errors_with_backend(df, error_types, config=None):
    """Use backend API for data annotation pipeline"""
    try:
        # Clean dataset first
        df_cleaned, message = clean_dataset(df)
        if df_cleaned is None:
            st.error(f"Dataset cleaning failed: {message}")
            return None
        
        st.info(message)
        
        # Convert dataframe to format expected by backend
        dataset = df_cleaned.to_dict('records')
        
        # Get configuration from UI if provided
        if config is None:
            config = {
                'llm_provider': 'openai',
                'model_name': 'gpt-4o-mini', 
                'base_url': None,
                'selected_error_types': error_types,
                'error_probabilities': [0.0, 0.7, 0.3],
                'include_key_points': True
            }
        
        # Process data annotation using original pipeline
        with st.spinner("Running data annotation pipeline with LLM..."):
            result = st.session_state.backend_client.start_data_annotation(
                dataset=dataset,
                llm_provider=config['llm_provider'],
                model_name=config['model_name'],
                base_url=config['base_url'],
                selected_error_types=config['selected_error_types'],
                error_probabilities=config['error_probabilities'],
                include_key_points=config['include_key_points']
            )
            
            if result and result.get('status') == 'completed':
                # Convert result back to dataframe
                annotated_data = result.get('annotated_dataset', [])
                if annotated_data:
                    return pd.DataFrame(annotated_data)
                else:
                    st.error("No annotated data returned")
                    return None
            else:
                st.error("Data annotation failed")
                return None
            
    except Exception as e:
        st.error(f"Data annotation failed: {str(e)}")
        # Fall back to mock data for demo purposes
        return generate_synthetic_errors_fallback(df, error_types)


def generate_synthetic_errors_fallback(df, error_types):
    """Fallback synthetic error generation"""
    error_df = df.copy()
    
    # Ensure Model_Answer column exists
    if 'Model_Answer' not in error_df.columns:
        if 'response' in error_df.columns:
            error_df['Model_Answer'] = error_df['response']
        elif 'Reference_Answer' in error_df.columns:
            error_df['Model_Answer'] = error_df['Reference_Answer']
        else:
            error_df['Model_Answer'] = error_df.iloc[:, 1]  # Use second column as fallback
    
    for idx, row in error_df.iterrows():
        selected_error = random.choice(error_types)
        original_answer = row['Model_Answer']
        
        if selected_error == 'Entity_Error':
            modified_answer = original_answer.replace('healthcare', 'education')
        elif selected_error == 'Negation':
            modified_answer = original_answer.replace('can', 'cannot')
        elif selected_error == 'Missing_Information':
            words = original_answer.split()
            modified_answer = ' '.join(words[:len(words)//2]) + "..."
        elif selected_error == 'Out_of_Reference':
            modified_answer = original_answer + " Additionally, this relates to quantum computing principles."
        else:  # Numerical_Error
            modified_answer = original_answer.replace('1)', '3)')
            
        error_df.at[idx, 'Model_Answer'] = modified_answer
        error_df.at[idx, 'Error_Type'] = selected_error
        error_df.at[idx, 'Original_Answer'] = original_answer
    
    return error_df


def start_agent_evaluation_with_backend(df, user_criteria, llm_judge):
    """Start Agent-based evaluation with backend"""
    try:
        backend_client = st.session_state.backend_client
        
        # Convert DataFrame to list of dicts
        dataset = df.to_dict('records')
        
        # Start agent evaluation
        task_id = backend_client.start_agent_evaluation(
            dataset=dataset,
            user_criteria=user_criteria,
            llm_provider="openai",
            model_name=llm_judge.lower().replace(" ", "-") if "openai" in llm_judge.lower() else "gpt-4o-mini"
        )
        
        # Wait for completion with progress updates
        st.info("🤖 AI agents are discussing optimal evaluation metrics...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        max_wait_time = 900  # 15 minutes for agent evaluation
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < max_wait_time:
            time.sleep(3)  # Check every 3 seconds
            check_count += 1
            
            try:
                progress = backend_client.get_agent_evaluation_progress(task_id)
                
                if progress.get("status") == "completed":
                    progress_bar.progress(100)
                    status_text.success("✅ Agent evaluation completed!")
                    break
                elif progress.get("status") == "error":
                    st.error(f"❌ Agent evaluation failed: {progress.get('error')}")
                    return None
                else:
                    # Show progress based on time elapsed
                    elapsed_time = time.time() - start_time
                    progress_pct = min(85, (elapsed_time / max_wait_time) * 100)
                    progress_bar.progress(int(progress_pct))
                    
                    stage = progress.get("progress", {}).get("stage", "processing")
                    elapsed_min = int(elapsed_time // 60)
                    elapsed_sec = int(elapsed_time % 60)
                    status_text.info(f"🔄 {stage.title()}... ({elapsed_min}m {elapsed_sec}s)")
                    
            except Exception as e:
                st.warning(f"Progress check {check_count}: {str(e)}")
                # Continue checking even if one progress check fails
                continue
        else:
            # Timeout reached
            return None
        
        # Get final result
        result = backend_client.get_agent_evaluation_result(task_id)
        return result
        
    except Exception as e:
        st.error(f"❌ Agent evaluation failed: {str(e)}")
        return None

def calculate_evaluation_scores_with_backend(df, llm_judge, metrics_weights):
    """Use backend API for evaluation"""
    try:
        # Convert dataset to backend expected format
        df_processed = df.copy()
        
        # Map column names to backend expected format
        column_mapping = {}
        for col in df_processed.columns:
            col_lower = col.lower().strip()
            if col_lower in ['question', 'query']:
                column_mapping[col] = 'Question'
            elif col_lower in ['response', 'answer', 'reference_answer', 'golden_answer', 'ground_truth']:
                column_mapping[col] = 'Reference_Answer'
            elif col_lower in ['model_answer', 'generated_answer', 'prediction']:
                column_mapping[col] = 'Model_Answer'
            elif col_lower in ['documents', 'context', 'doc', 'document', 'contexts']:
                column_mapping[col] = 'Context'
        
        # Apply column mapping
        if column_mapping:
            df_processed = df_processed.rename(columns=column_mapping)
        
        # Ensure required fields exist with flexible checking
        # Question field
        if 'Question' not in df_processed.columns:
            question_candidates = [col for col in df_processed.columns if col.lower() in ['question', 'query']]
            if question_candidates:
                df_processed['Question'] = df_processed[question_candidates[0]]
            else:
                st.error("❌ Dataset must contain a question field")
                return None
        
        # Reference Answer field
        if 'Reference_Answer' not in df_processed.columns:
            answer_candidates = [col for col in df_processed.columns 
                               if col.lower() in ['response', 'answer', 'reference_answer', 'golden_answer', 'ground_truth']]
            if answer_candidates:
                df_processed['Reference_Answer'] = df_processed[answer_candidates[0]]
            else:
                st.error("❌ Dataset must contain an answer field")
                return None
        
        # Model Answer field
        if 'Model_Answer' not in df_processed.columns:
            model_answer_candidates = [col for col in df_processed.columns 
                                     if col.lower() in ['model_answer', 'generated_answer', 'prediction']]
            if model_answer_candidates:
                df_processed['Model_Answer'] = df_processed[model_answer_candidates[0]]
            else:
                # Use Reference_Answer as fallback
                df_processed['Model_Answer'] = df_processed['Reference_Answer']
        
        # Context field (optional)
        if 'Context' not in df_processed.columns:
            context_candidates = [col for col in df_processed.columns 
                                if col.lower() in ['documents', 'context', 'doc', 'document', 'contexts']]
            if context_candidates:
                df_processed['Context'] = df_processed[context_candidates[0]]
        
        # Ensure key_points is in correct format before sending to backend
        if 'key_points' in df_processed.columns:
            def ensure_key_points_format(value):
                if pd.isna(value) or value is None:
                    return []
                if isinstance(value, str):
                    try:
                        import json
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return [str(item) for item in parsed]
                        else:
                            return [str(value)]
                    except:
                        if '|' in value:
                            return [item.strip() for item in value.split('|') if item.strip()]
                        elif ';' in value:
                            return [item.strip() for item in value.split(';') if item.strip()]
                        else:
                            return [str(value)]
                elif isinstance(value, list):
                    return [str(item) for item in value]
                else:
                    return [str(value)]
            
            df_processed['key_points'] = df_processed['key_points'].apply(ensure_key_points_format)
        
        # Convert to backend format
        dataset = df_processed.to_dict('records')
        
        # Get available evaluators from backend
        available_evaluators = st.session_state.backend_client.get_available_evaluators()
        evaluator_names = [ev['name'] for ev in available_evaluators]
        
        # Check dataset columns to filter out incompatible evaluators
        dataset_columns = set(df_processed.columns)
        
        # Define evaluators that require specific fields
        keypoint_evaluators = [
            'KeyPointCompletenessEvaluator', 
            'KeyPointHallucinationEvaluator', 
            'KeyPointIrrelevantEvaluator'
        ]
        
        # All evaluators should be compatible since we require key_points
        compatible_evaluators = evaluator_names
        
        # Convert metrics_weights to evaluator weights
        evaluator_weights = {}
        for metric_name, data in metrics_weights.items():
            # Map frontend metric names to backend evaluator names
            for ev_name in compatible_evaluators:
                if metric_name.lower().replace(' ', '').replace('_', '') in ev_name.lower().replace('_', ''):
                    evaluator_weights[ev_name] = data['weight']
                    break
        
        # Map LLM names to providers 
        llm_provider_map = {
            'OpenAI GPT-4o-mini': 'openai',
            'OpenAI GPT-4o': 'openai', 
            'OpenAI GPT-3.5-turbo': 'openai',
            'Qwen': 'qwen',
            'Deepseek': 'deepseek',
            'Distilled Qwen': 'qwen',
            'Mistral': 'mistral'
        }
        llm_provider = llm_provider_map.get(llm_judge, 'openai')
        
        # Map LLM selection to specific model names
        model_name_map = {
            'OpenAI GPT-4o-mini (Recommended)': 'gpt-4o-mini',
            'OpenAI GPT-4o': 'gpt-4o',
            'OpenAI GPT-3.5-turbo': 'gpt-3.5-turbo',
            'Qwen': 'qwen-plus',
            'Deepseek': 'deepseek-chat',
            'Distilled Qwen': 'qwen-turbo',
            'Mistral': 'mistral-7b-instruct'
        }
        model_name = model_name_map.get(llm_judge, 'gpt-4o-mini')
        

        
        # Start evaluation task
        task_id = st.session_state.backend_client.start_evaluation(
            dataset=dataset,
            selected_evaluators=list(evaluator_weights.keys()) if evaluator_weights else None,
            metric_weights=evaluator_weights if evaluator_weights else None,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        if task_id:
            # Poll for results with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_wait_time = 600  # 10 minutes for regular evaluation
            start_time = time.time()
            check_count = 0
            
            while time.time() - start_time < max_wait_time:
                try:
                    progress = st.session_state.backend_client.get_evaluation_progress(task_id)
                    check_count += 1
                    
                    if progress.get('status') == 'completed':
                        progress_bar.progress(100)
                        status_text.success("Evaluation completed!")
                        
                        result = st.session_state.backend_client.get_evaluation_result(task_id)
                        if result and 'dataset' in result:
                            return pd.DataFrame(result['dataset'])
                        else:
                            st.error("No evaluation results received")
                            return None
                    
                    elif progress.get('status') == 'error':
                        st.error(f"Evaluation failed: {progress.get('error', 'Unknown error')}")
                        return None
                    
                    else:
                        # Update progress with time information
                        prog_info = progress.get('progress', {})
                        elapsed_time = time.time() - start_time
                        elapsed_min = int(elapsed_time // 60)
                        elapsed_sec = int(elapsed_time % 60)
                        
                        if isinstance(prog_info, dict):
                            pct = prog_info.get('progress_percentage', 0)
                            progress_bar.progress(pct / 100)
                            current_eval = prog_info.get('current_evaluator', 'Processing...')
                            status_text.text(f"Evaluating: {current_eval} ({elapsed_min}m {elapsed_sec}s)")
                        else:
                            # Show progress based on elapsed time
                            progress_pct = min(80, (elapsed_time / max_wait_time) * 100)
                            progress_bar.progress(progress_pct / 100)
                            status_text.text(f"Evaluation in progress... ({elapsed_min}m {elapsed_sec}s)")
                
                except Exception as e:
                    st.warning(f"Progress check {check_count}: {str(e)}")
                    # Continue checking even if one progress check fails
                
                time.sleep(3)  # Check every 3 seconds
            
            return None
        else:
            st.error("Failed to start evaluation task")
            return None
            
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        # Fallback to mock evaluation
        return calculate_evaluation_scores_fallback(df, llm_judge, metrics_weights)


def calculate_evaluation_scores_fallback(df, llm_judge, metrics_weights):
    """Fallback evaluation with mock scores"""
    scored_df = df.copy()
    
    for idx, row in scored_df.iterrows():
        base_scores = {
            'Factual_Accuracy': random.uniform(6.0, 9.5),
            'Coherence': random.uniform(7.0, 9.8),
            'Relevance': random.uniform(7.5, 9.6),
            'Completeness': random.uniform(6.5, 9.0),
            'Citation_Quality': random.uniform(5.5, 8.5),
            'Clarity': random.uniform(7.5, 9.7),
            'Technical_Depth': random.uniform(6.0, 8.8)
        }
        
        # Lower scores for error data
        if 'Error_Type' in row and pd.notna(row.get('Error_Type')):
            for metric in base_scores:
                base_scores[metric] *= random.uniform(0.6, 0.8)
        
        # Calculate weighted total score
        weighted_scores = {}
        total_weighted_score = 0
        total_weight = 0
        
        for metric, score in base_scores.items():
            metric_name = metric.replace('_', ' ').title()
            if metric_name in metrics_weights:
                weight = metrics_weights[metric_name]['weight']
                weighted_scores[metric] = round(score, 2)
                total_weighted_score += score * weight
                total_weight += weight
        
        adams_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Add scores to dataframe
        for metric, score in weighted_scores.items():
            scored_df.at[idx, metric] = score
        
        scored_df.at[idx, 'ADAMS_Score'] = round(adams_score, 2)
        scored_df.at[idx, 'LLM_Judge'] = llm_judge
        scored_df.at[idx, 'Evaluation_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return scored_df


if st.session_state.metrics_data is None:
    st.session_state.metrics_data = default_metrics.copy()

# Check backend connection
backend_connected = st.session_state.backend_client.test_connection()

# Header
st.markdown('<h1 class="main-title">ADAMS</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 2rem;">Adaptive Domain-Aware Metric Selection</p>', unsafe_allow_html=True)

# Backend status indicator
if backend_connected:
    st.success("🟢 Backend API Connected")
else:
    st.warning("🟡 Backend API Unavailable - Check your backend server")

# Navigation tabs
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔧 Data Annotation", use_container_width=True, 
                 type="primary" if st.session_state.current_tab == 'error_generation' else "secondary"):
        st.session_state.current_tab = 'error_generation'
with col2:
    if st.button("📊 Evaluation", use_container_width=True,
                 type="primary" if st.session_state.current_tab == 'evaluation' else "secondary"):
        st.session_state.current_tab = 'evaluation'
with col3:
    if st.button("📈 History", use_container_width=True,
                 type="primary" if st.session_state.current_tab == 'history' else "secondary"):
        st.session_state.current_tab = 'history'

st.markdown("---")

# Tab 1: Data Annotation
if st.session_state.current_tab == 'error_generation':
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("## 🔧 Data Annotation Pipeline")
    st.markdown("Upload dataset and run the original annotation pipeline (Key Points, Mistake Distribution, Answer Generation)")
    
    # File upload and dataset processing
    st.markdown("### 📂 Dataset Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=['csv', 'json'],
        help="Support CSV and JSON formats. Expected columns: question, response, documents"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)
            
            # Clean the dataset
            df_cleaned, clean_message = clean_dataset(df)
            if df_cleaned is None:
                st.error(f"❌ Dataset cleaning failed: {clean_message}")
            else:
                st.session_state.original_dataset = df_cleaned
                st.success(f"✅ Successfully loaded and cleaned dataset")
                st.info(clean_message)
                
                # Show data preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df_cleaned.head(), use_container_width=True)
                
                # Show dataset statistics
                st.markdown("#### 📊 Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(df_cleaned))
                with col2:
                    st.metric("Columns", len(df_cleaned.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Store in session state for use in other functions
            st.session_state.uploaded_data = df_cleaned
            st.session_state.data_source = "uploaded_file"
            st.session_state.current_dataset = df_cleaned  # Store for evaluation use
            
        except Exception as e:
            st.error(f"❌ File loading failed: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to begin the annotation process")
    
    # Data annotation pipeline (only show if dataset is loaded)
    if st.session_state.original_dataset is not None:
        st.markdown("---")
        st.markdown("### 🎯 Data Annotation Pipeline")
        st.markdown("Run the original annotation pipeline: Key Points → Mistake Distribution → Answer Generation")
        
        # Check OpenAI API connectivity
        st.markdown("#### 🔗 API Status Check")
        col1, col2 = st.columns(2)
        with col1:
            if backend_connected:
                st.markdown("**Backend API**: <span class='status-online'>🟢 Connected</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Backend API**: <span class='status-offline'>🔴 Disconnected</span>", unsafe_allow_html=True)
        with col2:
            if st.button("🔍 Test OpenAI API", key="test_openai"):
                with st.spinner("Testing OpenAI API connectivity..."):
                    try:
                        import requests
                        test_response = requests.get("https://api.openai.com/v1/models", 
                                                   headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}, 
                                                   timeout=10)
                        if test_response.status_code == 200:
                            st.success("🟢 OpenAI API is accessible")
                        else:
                            st.error(f"🔴 OpenAI API returned status: {test_response.status_code}")
                    except Exception as e:
                        st.error(f"🔴 Cannot reach OpenAI API: {str(e)}")
                        st.info("💡 This might be due to network restrictions or firewall settings.")
        
        # Simple pipeline execution
        if st.button("🚀 Run Data Annotation Pipeline to add mistakes", use_container_width=True, type="primary"):
            with st.spinner("Running data annotation pipeline (MistakeAnswerGenerator)..."):
                if backend_connected:
                    # Use the original pipeline
                    annotated_df, message = run_original_annotation_pipeline(st.session_state.original_dataset)
                else:
                    st.warning("⚠️ Backend not connected. Please start the backend server.")
                    annotated_df, message = None, "Backend not available"
            
            if annotated_df is not None:
                st.session_state.error_dataset = annotated_df
                st.success(f"✅ {message}")
                
                # Show annotation statistics
                st.markdown("#### 📊 Pipeline Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(annotated_df))
                with col2:
                    st.metric("Original Samples", len(st.session_state.original_dataset))
                with col3:
                    new_cols = set(annotated_df.columns) - set(st.session_state.original_dataset.columns)
                    st.metric("New Columns", len(new_cols))
                
                # Show column details
                if new_cols:
                    st.markdown(f"**New columns added**: {', '.join(sorted(new_cols))}")
                
                # Show pipeline stages completed
                st.markdown("#### ✅ Pipeline Stages Completed")
                stages = ["NumMistakesAnnotator", "MistakeDistributionAnnotator", "MistakeAnswerGenerator"]
                for i, stage in enumerate(stages, 1):
                    st.markdown(f"{i}. **{stage}** ✓")
                    
            else:
                st.error(f"❌ Pipeline execution failed: {message}")
                
                # Provide troubleshooting guidance
                if "OpenAI API key" in message or "Invalid API key" in message:
                    st.markdown("#### 🔧 Troubleshooting Guide")
                    st.markdown("""
                    **OpenAI API Key Issues:**
                    1. Make sure you have a valid OpenAI API key
                    2. Check that the OPENAI_API_KEY environment variable is set
                    3. Verify your API key has sufficient credits
                    4. Ensure your network can access api.openai.com
                    
                    **Network Issues:**
                    - Check your internet connection
                    - Verify firewall settings allow access to OpenAI API
                    - Try using a VPN if access is restricted in your region
                    """)
    
    # Display annotated dataset (only if available)
    if st.session_state.error_dataset is not None:
        st.markdown("---")
        st.markdown("### 📋 Annotated Dataset")
        
        # Dataset preview with pagination
        st.markdown("#### 👀 Dataset Preview")
        preview_rows = st.slider("Number of rows to display", 5, 50, 10, key="preview_rows")
        st.dataframe(st.session_state.error_dataset.head(preview_rows), use_container_width=True)
        
        # Download options
        st.markdown("#### 💾 Download Options")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.error_dataset.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"annotated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = st.session_state.error_dataset.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"annotated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Evaluation
elif st.session_state.current_tab == 'evaluation':
    st.markdown("## 📊 Evaluation & Weight Adjustment")
    st.markdown("Evaluate datasets and adjust evaluation metric weights")
    
    # Use single column layout for better display
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    
    # Dataset upload
    st.markdown("### 📂 Upload Dataset for Evaluation")
    
    uploaded_eval_file = st.file_uploader(
        "Upload evaluation dataset",
        type=['csv', 'json'],
        help="Required fields: question, response, documents, key_points",
        key="evaluation_file_upload"
    )
    
    current_dataset = None
    
    if uploaded_eval_file is not None:
        try:
            if uploaded_eval_file.name.endswith('.csv'):
                current_dataset = pd.read_csv(uploaded_eval_file)
            else:
                current_dataset = pd.read_json(uploaded_eval_file)
            
            # Validate required columns
            required_cols = ['question', 'response', 'documents', 'key_points']
            missing_cols = [col for col in required_cols if col not in current_dataset.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                current_dataset = None
            else:
                # Process key_points field to ensure correct format
                def process_key_points(value):
                    if pd.isna(value) or value is None:
                        return []
                    if isinstance(value, str):
                        try:
                            # Try to parse as JSON
                            import json
                            parsed = json.loads(value)
                            if isinstance(parsed, list):
                                return [str(item) for item in parsed]
                            else:
                                return [str(value)]
                        except json.JSONDecodeError:
                            # Split by common delimiters
                            if '|' in value:
                                return [item.strip() for item in value.split('|') if item.strip()]
                            elif ';' in value:
                                return [item.strip() for item in value.split(';') if item.strip()]
                            else:
                                return [str(value)]
                    elif isinstance(value, list):
                        return [str(item) for item in value]
                    else:
                        return [str(value)]
                
                current_dataset['key_points'] = current_dataset['key_points'].apply(process_key_points)
                
                # Save to session state
                st.session_state.current_eval_dataset = current_dataset
                st.success(f"✅ Dataset loaded successfully: {len(current_dataset)} samples")
        
        except Exception as e:
            st.error(f"❌ File loading failed: {str(e)}")
            current_dataset = None
    
    # Use session state dataset if available and no new file uploaded
    elif 'current_eval_dataset' in st.session_state and st.session_state.current_eval_dataset is not None:
        current_dataset = st.session_state.current_eval_dataset
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.info("📋 Using previously uploaded dataset")
        with col_clear:
            if st.button("🗑️ Clear"):
                del st.session_state.current_eval_dataset
                # Clear related evaluation states as well
                if 'current_evaluation_result' in st.session_state:
                    del st.session_state.current_evaluation_result
                if 'agent_selected_metrics' in st.session_state:
                    del st.session_state.agent_selected_metrics
                if 'agent_discussion_summary' in st.session_state:
                    del st.session_state.agent_discussion_summary
                st.success("✅ Dataset cleared")
    
    # Show preview of selected dataset
    if current_dataset is not None:
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(current_dataset.head(3), use_container_width=True)
            st.metric("Total Samples", len(current_dataset))
    
    # Agent Evaluation Configuration
    st.markdown("### 🤖 Agent-based Dynamic Evaluation")
    st.info("🧠 AI agents will automatically analyze your data and select the most suitable evaluation metrics with optimal weights.")
    
    # Simplified LLM selection - only Evaluation LLM
    st.markdown("### 🤖 Select Evaluation LLM")
    
    # Evaluation LLM (Agent LLM will use the same model)
    eval_llm_options = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
    if 'eval_llm' not in st.session_state:
        st.session_state.eval_llm = 'gpt-4o-mini'
        
    st.session_state.eval_llm = st.selectbox(
        "Select LLM model for evaluation:",
        eval_llm_options,
        index=eval_llm_options.index(st.session_state.eval_llm) if st.session_state.eval_llm in eval_llm_options else 0,
        help="LLM used for both agent discussion and metric evaluation"
    )
    
    # Initialize user_criteria if not present
    if 'user_criteria' not in st.session_state:
        st.session_state.user_criteria = """Please help build evaluate metrics for chatbot run by technical safety BC(TSBC), Here are the metrics and their weights
- FactualAccuracyEvaluator: 20%
- FactualCorrectnessEvaluator: 15%
- KeyPointCompletenessEvaluator: 20%
- KeyPointHallucinationEvaluator: 15%
- ContextRelevanceEvaluator: 10%
- CoherenceEvaluator: 10%
- EngagementEvaluator: 10%"""

    
    # Start Agent-based Evaluation
    if current_dataset is not None and st.button("🚀 Start Agent-based Evaluation", use_container_width=True, type="primary"):
        # Use fixed user criteria (matching agent_e2e.py format)
        user_criteria = """Please help build evaluate metrics for chatbot run by technical safety BC(TSBC), Here are the metrics and their weights
- FactualAccuracyEvaluator: 20%
- FactualCorrectnessEvaluator: 15%
- KeyPointCompletenessEvaluator: 20%
- KeyPointHallucinationEvaluator: 15%
- ContextRelevanceEvaluator: 10%
- CoherenceEvaluator: 10%
- EngagementEvaluator: 10%"""
        
        # Save user_criteria to session_state for later access
        st.session_state.user_criteria = user_criteria
        
        with st.spinner(f"🤖 Starting Agent-based evaluation with {st.session_state.eval_llm}..."):
            agent_result = None
            try:
                if backend_connected:
                    # Use the backend client with new parameters
                    dataset = current_dataset.to_dict('records')
                    task_id = st.session_state.backend_client.start_agent_evaluation(
                        dataset=dataset,
                        user_criteria=user_criteria,
                        llm_provider="openai",
                        model_name=st.session_state.eval_llm
                    )
                    
                    # Wait for completion with progress updates
                    st.info("🤖 AI agents are discussing optimal evaluation metrics...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    max_wait_time = 900  # 15 minutes for agent evaluation
                    start_time = time.time()
                    check_count = 0
                    
                    while time.time() - start_time < max_wait_time:
                        time.sleep(3)  # Check every 3 seconds
                        check_count += 1
                        
                        try:
                            progress = st.session_state.backend_client.get_agent_evaluation_progress(task_id)
                            
                            if progress.get("status") == "completed":
                                progress_bar.progress(100)
                                status_text.success("✅ Agent evaluation completed!")
                                break
                            elif progress.get("status") == "error":
                                st.error(f"❌ Agent evaluation failed: {progress.get('error')}")
                                agent_result = None
                                break
                            else:
                                # Show progress based on time elapsed
                                elapsed_time = time.time() - start_time
                                progress_pct = min(85, (elapsed_time / max_wait_time) * 100)
                                progress_bar.progress(int(progress_pct))
                                
                                stage = progress.get("progress", {}).get("stage", "processing")
                                elapsed_min = int(elapsed_time // 60)
                                elapsed_sec = int(elapsed_time % 60)
                                status_text.info(f"🔄 {stage.title()}... ({elapsed_min}m {elapsed_sec}s)")
                                
                        except Exception as e:
                            st.warning(f"Progress check {check_count}: {str(e)}")
                            # Continue checking even if one progress check fails
                            continue
                    else:
                        # Timeout reached
                        agent_result = None
                    
                    # Try to get final result if we completed successfully or hit timeout
                    if time.time() - start_time < max_wait_time:
                        # Completed successfully, get the result
                        try:
                            agent_result = st.session_state.backend_client.get_agent_evaluation_result(task_id)
                        except Exception as e:
                            st.error(f"Failed to get agent evaluation result: {str(e)}")
                            agent_result = None
                        
                else:
                    st.error("❌ Backend connection required for Agent-based evaluation")
                    agent_result = None
            except Exception as e:
                st.error(f"❌ Agent evaluation failed: {str(e)}")
                agent_result = None
            
            
            if agent_result:
                # Display agent discussion results  
                st.success("✅ Agent-based evaluation completed!")
                
                # Handle different agent result formats
                if agent_result.get("status") == "success":
                    st.success("🎉 Agent evaluation completed successfully!")
                    
                    # Display selected metrics
                    selected_metrics = agent_result.get("selected_metrics", {})
                    if selected_metrics:
                        st.subheader("📊 Agent-Selected Metrics")
                        
                        # Create a DataFrame for better display
                        metrics_df = pd.DataFrame([
                            {"Evaluator": k, "Weight": f"{v:.2%}"} 
                            for k, v in selected_metrics.items()
                        ])
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Store metrics in session state for slider updates
                        st.session_state.agent_selected_metrics = selected_metrics
                    
                    # Store and display discussion summary with enhanced UI
                    discussion_summary = agent_result.get("discussion_summary", "")
                    if discussion_summary:
                        # Store in session state for use in other functions
                        st.session_state.agent_discussion_summary = discussion_summary
                        
                        # Display enhanced discussion summary with visual elements
                        display_discussion_summary_with_sliders(discussion_summary, selected_metrics)
                    
                    # Display evaluation results if available
                    evaluation_result = agent_result.get("evaluation_result")
                    if evaluation_result and agent_result.get("has_evaluation_data"):
                        # Store evaluation result in session state for persistent weight adjustment
                        st.session_state.current_evaluation_result = evaluation_result
                        
                        # Use the new metric scores display with weight adjustment (final score shown inside)
                        display_metric_scores_with_adjustment(evaluation_result, selected_metrics)
                        
                        # Show sample evaluated data
                        processed_dataset = evaluation_result.get("processed_dataset", [])
                        if processed_dataset:
                            st.subheader("📄 Sample Evaluated Data")
                            st.write(f"**Total Samples Evaluated:** {len(processed_dataset)}")
                            
                            # Show sample data with scores
                            if len(processed_dataset) > 0:
                                with st.expander("🔍 Sample Evaluated Data", expanded=False):
                                    sample_df = pd.DataFrame(processed_dataset[:3])  # Show first 3 samples
                                    st.dataframe(sample_df, use_container_width=True)
                        

                
                    else:
                        st.info("💡 Agent negotiation completed. Evaluation results not available in this response.")
                        st.write("**Note:** You can use the selected metrics to run a separate evaluation.")
                        
                        # Add evaluation button when metrics are available but no evaluation results
                        if selected_metrics:
                            st.markdown("---")
                            col1, col2 = st.columns([1, 1])
                            with col2:
                                if st.button("🔍 Run Evaluation with Selected Metrics", type="primary", use_container_width=True):
                                    if hasattr(st.session_state, 'current_dataset') and st.session_state.current_dataset:
                                        run_evaluation_with_backend(
                                            st.session_state.current_dataset, 
                                            st.session_state.eval_llm, 
                                            selected_metrics
                                        )
                                    else:
                                        st.error("❌ No dataset available for evaluation")
                
                elif agent_result.get("status") == "error":
                    st.error(f"❌ Agent evaluation failed: {agent_result.get('error', 'Unknown error')}")
                    
                    # Still show selected metrics if available (fallback metrics)
                    selected_metrics = agent_result.get("selected_metrics", {})
                    if selected_metrics:
                        st.subheader("🔄 Fallback Metrics")
                        st.info("Using default metrics due to agent discussion failure.")
                        
                        metrics_df = pd.DataFrame([
                            {"Evaluator": k, "Weight": f"{v:.2%}"} 
                            for k, v in selected_metrics.items()
                        ])
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Store fallback metrics in session state
                        st.session_state.agent_selected_metrics = selected_metrics
                    
                    # Show error discussion summary
                    discussion_summary = agent_result.get("discussion_summary", "")
                    if discussion_summary:
                        with st.expander("❌ Error Details", expanded=False):
                            st.text(discussion_summary)
                    
                    # Show evaluation results if somehow available despite error
                    evaluation_result = agent_result.get("evaluation_result")
                    if evaluation_result and agent_result.get("has_evaluation_data"):
                        st.subheader("📊 Partial Evaluation Results")
                        final_score = evaluation_result.get("final_score")
                        if final_score is not None:
                            st.metric("Final Score (Partial)", f"{final_score:.3f}")
                
                else:
                    st.warning(f"⚠️ Unknown status: {agent_result.get('status')}")
            
            else:
                # Agent evaluation failed or timed out - show simplified error
                st.error("❌ Agent evaluation failed, please try again or check your API configuration")
                
                # Simple retry button
                if st.button("🔄 Retry Agent Evaluation", use_container_width=True, type="primary"):
                    st.rerun()

    # Display current evaluation results with weight adjustment if available
    if ('current_evaluation_result' in st.session_state and 
        'agent_selected_metrics' in st.session_state and 
        st.session_state.current_evaluation_result is not None and
        st.session_state.agent_selected_metrics):
        
        st.markdown("---")
        st.subheader("📊 Current Evaluation Results")
        display_metric_scores_with_adjustment(
            st.session_state.current_evaluation_result, 
            st.session_state.agent_selected_metrics
        )
        
        # Display discussion summary if available
        if 'agent_discussion_summary' in st.session_state:
            display_discussion_summary_with_sliders(
                st.session_state.agent_discussion_summary, 
                st.session_state.agent_selected_metrics
            )
    
    # Add evaluation status summary if there are previous results
    elif st.session_state.evaluation_history and len(st.session_state.evaluation_history) > 0:
        st.markdown("---")
        st.subheader("📈 Previous Evaluation Summary")
        
        # Get the latest evaluation result
        latest = st.session_state.evaluation_history[-1]
        
        # Check if the latest evaluation has score data
        if 'data' in latest and len(latest['data']) > 0:
            eval_df = latest['data']
            
            # Find score columns - try different possible score column names
            score_columns = [col for col in eval_df.columns if 'score' in col.lower() or 'adams' in col.lower()]
            
            if score_columns:
                # Use the first available score column
                score_col = score_columns[0]
                avg_score = eval_df[score_col].mean()
                
                # Display current evaluation result
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="📊 Latest Evaluation Score",
                        value=f"{avg_score:.3f}",
                        help=f"Based on {len(eval_df)} samples using {latest['llm_judge']}"
                    )
                
                # Show evaluation details in expander
                with st.expander("📋 Latest Evaluation Details", expanded=False):
                    st.markdown(f"**Model**: {latest['llm_judge']}")
                    st.markdown(f"**Samples**: {latest['sample_count']}")
                    st.markdown(f"**Time**: {latest['timestamp']}")
                    
                    # Show available metrics
                    st.markdown("**Available Metrics:**")
                    metric_cols = [col for col in eval_df.columns if col not in ['question', 'response', 'documents', 'generated_answer', 'key_points']]
                    for col in metric_cols[:5]:  # Show first 5 metrics
                        if eval_df[col].dtype in ['float64', 'int64']:
                            avg_val = eval_df[col].mean()
                            st.markdown(f"• **{col}**: {avg_val:.3f}")
            else:
                st.info("📊 Previous evaluation completed but no score columns found")
        else:
            # Show just basic info if no score data
            st.info(f"📊 Latest evaluation: {latest['name']} ({latest['timestamp']})")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: History Comparison
elif st.session_state.current_tab == 'history':
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Evaluation History & Comparison")
    st.markdown("View evaluation history and perform detailed comparison analysis")
    
    if len(st.session_state.evaluation_history) == 0:
        st.warning("⚠️ No evaluation history found. Please complete evaluation in the Evaluation tab first.")
        if st.button("📊 Go to Evaluation Page", use_container_width=True, type="primary"):
            st.session_state.current_tab = 'evaluation'
            st.rerun()
    
    else:
        # Display all history records
        st.markdown("### 📋 Evaluation History Records")
        
        history_data = []
        for record in st.session_state.evaluation_history:
            # Find available score columns
            if 'data' in record and len(record['data']) > 0:
                eval_df = record['data']
                score_columns = [col for col in eval_df.columns if 'score' in col.lower() or 'adams' in col.lower()]
                
                if score_columns:
                    avg_score = eval_df[score_columns[0]].mean()
                    avg_score_str = f"{avg_score:.2f}"
                else:
                    avg_score_str = "N/A"
            else:
                avg_score_str = "N/A"
                
            history_data.append({
                'ID': record['id'],
                'Name': record['name'],
                'Dataset Type': record['dataset_type'],
                'LLM Judge': record['llm_judge'],
                'Sample Count': record['sample_count'],
                'Average Score': avg_score_str,
                'Evaluation Time': record['timestamp']
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Comparison analysis section
        if len(st.session_state.evaluation_history) >= 2:
            st.markdown("### ⚖️ Comparison Analysis")
            
            col1, col2 = st.columns(2)
            
            eval_options = [(i, record['name']) for i, record in enumerate(st.session_state.evaluation_history)]
            
            with col1:
                st.markdown("#### 🔵 Evaluation Record A")
                selected_a_idx = st.selectbox(
                    "Select first evaluation record:",
                    options=[opt[0] for opt in eval_options],
                    format_func=lambda x: eval_options[x][1],
                    key="eval_a_select"
                )
                
                if selected_a_idx is not None:
                    eval_a = st.session_state.evaluation_history[selected_a_idx]
                    st.session_state.comparison_selection['eval_a'] = eval_a
                    
                    st.info(f"""
                    **Record Information:**
                    • **Name**: {eval_a['name']}
                    • **LLM Judge**: {eval_a['llm_judge']}
                    • **Sample Count**: {eval_a['sample_count']}
                    • **Evaluation Time**: {eval_a['timestamp']}
                    """)
            
            with col2:
                st.markdown("#### 🔴 Evaluation Record B")
                available_b_options = [opt for opt in eval_options if opt[0] != selected_a_idx]
                
                if available_b_options:
                    selected_b_idx = st.selectbox(
                        "Select second evaluation record:",
                        options=[opt[0] for opt in available_b_options],
                        format_func=lambda x: next(opt[1] for opt in eval_options if opt[0] == x),
                        key="eval_b_select"
                    )
                    
                    if selected_b_idx is not None:
                        eval_b = st.session_state.evaluation_history[selected_b_idx]
                        st.session_state.comparison_selection['eval_b'] = eval_b
                        
                        st.info(f"""
                        **Record Information:**
                        • **Name**: {eval_b['name']}
                        • **LLM Judge**: {eval_b['llm_judge']}
                        • **Sample Count**: {eval_b['sample_count']}
                        • **Evaluation Time**: {eval_b['timestamp']}
                        """)
            
            # Display detailed comparison
            if (st.session_state.comparison_selection['eval_a'] is not None and 
                st.session_state.comparison_selection['eval_b'] is not None):
                
                eval_a = st.session_state.comparison_selection['eval_a']
                eval_b = st.session_state.comparison_selection['eval_b']
                df_a = eval_a['data']
                df_b = eval_b['data']
                
                st.markdown("### 📊 Detailed Comparison Analysis")
                
                # Score comparison
                col1, col2, col3, col4 = st.columns(4)
                
                # Find available score columns for both datasets
                score_columns_a = [col for col in df_a.columns if 'score' in col.lower() or 'adams' in col.lower()]
                score_columns_b = [col for col in df_b.columns if 'score' in col.lower() or 'adams' in col.lower()]
                
                if score_columns_a and score_columns_b:
                    avg_a = df_a[score_columns_a[0]].mean()
                    avg_b = df_b[score_columns_b[0]].mean()
                    score_diff = avg_a - avg_b
                else:
                    avg_a = 0.0
                    avg_b = 0.0
                    score_diff = 0.0
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-display">
                        <div class="metric-value">{avg_a:.2f}</div>
                        <div class="metric-name">Record A Average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-display">
                        <div class="metric-value">{avg_b:.2f}</div>
                        <div class="metric-name">Record B Average</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    color = "#00f5ff" if score_diff >= 0 else "#ff006e"
                    st.markdown(f"""
                    <div class="metric-display">
                        <div class="metric-value" style="color: {color};">{score_diff:+.2f}</div>
                        <div class="metric-name">Score Difference</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    better_eval = eval_a['name'] if score_diff > 0 else eval_b['name'] if score_diff < 0 else "Tie"
                    st.markdown(f"""
                    <div class="metric-display">
                        <div class="metric-value" style="font-size: 1.5rem;">{"🏆" if score_diff != 0 else "🤝"}</div>
                        <div class="metric-name">{'Better Record' if score_diff != 0 else 'Result'}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metric comparison
                st.markdown("#### 🧬 Detailed Metric Comparison")
                
                metric_columns = [col for col in df_a.columns if col.replace('_', ' ').title() in default_metrics.keys()]
                
                if metric_columns:
                    comparison_data = []
                    for metric in metric_columns:
                        if metric in df_b.columns:
                            mean_a = df_a[metric].mean()
                            mean_b = df_b[metric].mean()
                            difference = mean_a - mean_b
                            
                            comparison_data.append({
                                'Metric': metric.replace('_', ' ').title(),
                                f'{eval_a["llm_judge"]} (A)': round(mean_a, 2),
                                f'{eval_b["llm_judge"]} (B)': round(mean_b, 2),
                                'Difference': round(difference, 2),
                                'Better': eval_a['llm_judge'] if difference > 0 else eval_b['llm_judge'] if difference < 0 else 'Tie'
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Download comparison report
                        report_data = {
                            "Comparison Summary": {
                                "Evaluation Record A": eval_a['name'],
                                "Evaluation Record B": eval_b['name'],
                                "Average Score Difference": score_diff,
                                "Better Record": better_eval,
                                "Comparison Time": datetime.now().strftime(
                                 "%Y-%m-%d %H:%M:%S")
                            },
                            "Detailed Metric Comparison": comparison_data
                        }
                        
                        st.download_button(
                            label="📊 Download Comparison Report",
                            data=json.dumps(report_data, ensure_ascii=False, indent=2),
                            file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
        
        else:
            st.info("💡 At least 2 evaluation records are required for comparison analysis")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🧠 ADAMS Console")
    st.markdown("**Version**: 3.0.0")
    st.markdown("**Status**: <span class='status-online'>✅ Online</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Current Session")
    tab_display_names = {
        'error_generation': 'Data Annotation',
        'evaluation': 'Evaluation', 
        'history': 'History'
    }
    current_tab_name = tab_display_names.get(st.session_state.current_tab, st.session_state.current_tab)
    st.markdown(f"**Current Tab**: {current_tab_name}")
    st.markdown(f"**LLM Judge**: {st.session_state.selected_llm}")
    
    # Dataset status
    st.markdown("### 📂 Dataset Status")
    if st.session_state.original_dataset is not None:
        st.markdown(f"✅ **Original Dataset**: {len(st.session_state.original_dataset)} samples")
    else:
        st.markdown("❌ **Original Dataset**: Not loaded")
    
    if st.session_state.error_dataset is not None:
        st.markdown(f"✅ **Annotated Dataset**: {len(st.session_state.error_dataset)} samples")
    else:
        st.markdown("❌ **Annotated Dataset**: Not generated")
    
    # Evaluation history
    st.markdown("### 📈 Evaluation History")
    st.markdown(f"**History Count**: {len(st.session_state.evaluation_history)}")
    
    if st.session_state.evaluation_history:
        latest = st.session_state.evaluation_history[-1]
        st.markdown(f"**Latest Evaluation**: {latest['name'][:20]}...")
        
        # Find available score columns
        if 'data' in latest and len(latest['data']) > 0:
            eval_df = latest['data']
            score_columns = [col for col in eval_df.columns if 'score' in col.lower() or 'adams' in col.lower()]
            
            if score_columns:
                avg_score = eval_df[score_columns[0]].mean()
                st.markdown(f"**Average Score**: {avg_score:.2f}")
            else:
                st.markdown("**Average Score**: N/A")
        else:
            st.markdown("**Average Score**: N/A")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in ['original_dataset', 'error_dataset', 'evaluation_history', 'comparison_selection', 'current_eval_dataset']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.metrics_data = default_metrics.copy()
        st.session_state.current_tab = 'error_generation'
        st.rerun()
    
    # Usage guide
    st.markdown("---")
    st.markdown("### 📖 Usage Guide")
    st.markdown("""
    **Basic Workflow:**
    1. **Data Annotation**: Upload dataset, generate synthetic errors and annotate them
    2. **Evaluation**: Configure weights, calculate evaluation scores
    3. **History**: Compare different evaluation results
    
    **Tips**: 
    - Support multiple error type combinations
    - Weights affect final scores in real-time
    - History records can export analysis reports
    """)

def run_evaluation_with_backend(dataset, llm_judge, selected_metrics=None):
    """Run evaluation using backend API with selected metrics"""
    
    if not selected_metrics and 'agent_selected_metrics' in st.session_state:
        selected_metrics = st.session_state.agent_selected_metrics
    
    if not selected_metrics:
        st.error("❌ No metrics selected for evaluation")
        return
    
    try:
        with st.spinner("🔄 Running evaluation with selected metrics..."):
            # Create metrics configuration for the pipeline
            metrics_config = [
                {
                    "evaluator_name": evaluator_name,
                    "weight": weight,
                    "enabled": True
                }
                for evaluator_name, weight in selected_metrics.items()
            ]
            
            # Call the backend evaluation pipeline
            client = get_backend_client()
            result = client.run_evaluation_pipeline(
                dataset=dataset,
                llm_provider="openai",
                model_name=llm_judge,
                metrics_config=metrics_config
            )
            
            if result.get("status") == "success":
                st.success("✅ Evaluation completed successfully!")
                
                # Display results
                st.subheader("📊 Evaluation Results")
                
                # Show final score
                final_score = result.get("final_score")
                if final_score is not None:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="🎯 Final Score", 
                            value=f"{final_score:.3f}",
                            help="Weighted average across all metrics"
                        )
                
                # Show individual metric results
                metrics_results = result.get("metrics", [])
                if metrics_results:
                    st.subheader("📋 Individual Metric Results")
                    
                    for metric in metrics_results:
                        evaluator_name = metric.get("evaluator_name", "Unknown")
                        avg_score = metric.get("average_score", 0)
                        weight = selected_metrics.get(evaluator_name, 0)
                        
                        with st.expander(f"📊 {evaluator_name}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Score", f"{avg_score:.3f}")
                            with col2:
                                st.metric("Weight", f"{weight:.2%}")
                            with col3:
                                st.metric("Contribution", f"{avg_score * weight:.3f}")
                
                # Show dataset results summary
                processed_dataset = result.get("processed_dataset", [])
                if processed_dataset:
                    st.subheader("📄 Dataset Results")
                    st.write(f"**Samples Evaluated:** {len(processed_dataset)}")
                    
                    # Show sample results
                    with st.expander("🔍 Sample Results", expanded=False):
                        sample_df = pd.DataFrame(processed_dataset[:3])
                        st.dataframe(sample_df, use_container_width=True)
                
            else:
                st.error(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")


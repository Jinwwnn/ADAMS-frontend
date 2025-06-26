import streamlit as st
import time
import json
import pandas as pd
import io
import random

# Configure page
st.set_page_config(
    page_title="ADAMS - RAG Evaluation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic styling
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
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
        background: linear-gradient(45deg, #ff006e, #8338ec, #3a86ff, #06ffa5);
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
    
    /* Cyber cards */
    .cyber-card {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 245, 255, 0.1);
    }
    
    /* Neon text */
    .neon-text {
        color: #00f5ff;
        text-shadow: 0 0 10px #00f5ff;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-display {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        color: #00f5ff;
        text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
    }
    
    .metric-name {
        font-size: 0.9rem;
        color: #b8bcc8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Score display */
    .score-display {
        text-align: center;
        padding: 2rem;
        background: rgba(0, 245, 255, 0.1);
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        color: #00f5ff;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Comment section */
    .comment-section {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'dataset_processed' not in st.session_state:
    st.session_state.dataset_processed = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = 'Qwen'
if 'reviewer_comments' not in st.session_state:
    st.session_state.reviewer_comments = {}
if 'processed_datasets_history' not in st.session_state:
    st.session_state.processed_datasets_history = []
if 'comparison_selection' not in st.session_state:
    st.session_state.comparison_selection = {'dataset_a': None, 'dataset_b': None}

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

def process_uploaded_dataset(uploaded_file, selected_llm):
    """Process uploaded dataset and add ADAMS scores and metrics"""
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            return None
        
        # Ensure required columns exist
        required_columns = ['Question', 'Reference_Answer', 'Model_Answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Add ADAMS processing results
        processed_data = []
        for _, row in df.iterrows():
            # Simulate ADAMS metric evaluation with some randomness but realistic scores
            base_scores = {
                'Factual_Accuracy': random.uniform(7.5, 9.5),
                'Coherence': random.uniform(8.0, 9.8),
                'Relevance': random.uniform(8.2, 9.6),
                'Completeness': random.uniform(7.0, 9.0),
                'Citation_Quality': random.uniform(6.5, 8.5),
                'Clarity': random.uniform(8.5, 9.7),
                'Technical_Depth': random.uniform(7.2, 8.8)
            }
            
            # Calculate overall ADAMS score (weighted average)
            weights = [0.9, 0.8, 0.85, 0.7, 0.75, 0.7, 0.7]
            adams_score = sum(score * weight for score, weight in zip(base_scores.values(), weights)) / sum(weights)
            
            processed_row = {
                'Question': row['Question'],
                'Reference_Answer': row['Reference_Answer'], 
                'Model_Answer': row['Model_Answer'],
                'ADAMS_Score': round(adams_score, 2),
                'LLM_Judge': selected_llm,
                'Factual_Accuracy': round(base_scores['Factual_Accuracy'], 2),
                'Coherence': round(base_scores['Coherence'], 2),
                'Relevance': round(base_scores['Relevance'], 2),
                'Completeness': round(base_scores['Completeness'], 2),
                'Citation_Quality': round(base_scores['Citation_Quality'], 2),
                'Clarity': round(base_scores['Clarity'], 2),
                'Technical_Depth': round(base_scores['Technical_Depth'], 2),
                'Processing_Timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'Original_Data': False
            }
            processed_data.append(processed_row)
            
        return processed_data
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

if st.session_state.metrics_data is None:
    st.session_state.metrics_data = default_metrics.copy()

# Header
st.markdown('<h1 class="main-title">ADAMS</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #b8bcc8; margin-bottom: 2rem;">Adaptive Domain-Aware Metric Selection</p>', unsafe_allow_html=True)

# Navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 Dataset Upload", use_container_width=True, type="primary"):
        st.session_state.page = 'upload'
with col2:
    if st.button("📋 Dataset Review", use_container_width=True, type="primary"):
        st.session_state.page = 'dataset'
with col3:
    if st.button("🎛️ Configuration", use_container_width=True, type="primary"):
        st.session_state.page = 'config'
with col4:
    if st.button("⚖️ Compare", use_container_width=True, type="primary"):
        st.session_state.page = 'compare'

# Page 1: Upload
if st.session_state.page == 'upload':
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("## 🧠 Dataset Processing")
    st.markdown("Upload your RAG outputs for multi-agent evaluation analysis")
    
    # Download sample dataset section
    st.markdown("### 📥 Need a Sample Dataset?")
    sample_csv_data = """Question,Reference_Answer,Model_Answer
"What are the key benefits of using RAG systems in healthcare applications?","RAG systems provide up-to-date medical information, reduce hallucinations, ensure compliance, and enable personalized care.","RAG systems in healthcare offer several key benefits: 1) Access to up-to-date medical research and guidelines, 2) Reduced hallucination through grounded responses, 3) Compliance with regulatory requirements through traceable sources, and 4) Personalized patient care through dynamic information retrieval."
"How do transformer architectures handle long sequences?","Transformers use attention mechanisms but face quadratic complexity with sequence length, leading to various optimization techniques.","Transformer architectures handle long sequences through self-attention mechanisms, though they face computational challenges due to quadratic complexity. Modern approaches include attention optimization, sparse attention patterns, and hierarchical processing to manage memory and computational requirements effectively."
"What is the difference between supervised and unsupervised learning?","Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data.","Supervised learning algorithms learn from labeled training data to make predictions on new data, while unsupervised learning discovers hidden patterns and structures in data without labels, such as clustering and dimensionality reduction techniques."
"Explain the concept of transfer learning in deep learning.","Transfer learning involves using pre-trained models and adapting them to new tasks, reducing training time and data requirements.","Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a related task. This approach leverages knowledge gained from pre-trained models, significantly reducing training time and computational resources while often achieving better performance."
"What are the main challenges in implementing RAG systems?","Key challenges include retrieval quality, context length limitations, computational costs, and maintaining consistency between retrieved and generated content.","The main challenges in implementing RAG systems include: 1) Ensuring high-quality and relevant document retrieval, 2) Managing context length limitations in language models, 3) Balancing computational costs with performance, 4) Maintaining consistency between retrieved information and generated responses, and 5) Handling conflicting information from multiple sources."""
    
    st.download_button(
        label="📥 Download Sample Dataset (CSV)",
        data=sample_csv_data,
        file_name="sample_rag_dataset.csv",
        mime="text/csv",
        help="Download this sample dataset to test the ADAMS interface"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Drop your data into the evaluation system",
        type=['csv', 'json'],
        help="Supports CSV, JSON formats • Max 200MB"
    )
    
    # LLM Judge Selection
    st.markdown("### 🤖 Select LLM Judge")
    llm_options = ['Qwen', 'Deepseek', 'Distilled Qwen', 'Mistral', 'LLaMA 3.1']
    st.session_state.selected_llm = st.selectbox(
        "Choose the LLM judge for evaluation:",
        llm_options,
        index=llm_options.index(st.session_state.selected_llm)
    )
    
    if uploaded_file is not None:
        # Show uploaded file info with delete option
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(f"✅ File uploaded: **{uploaded_file.name}** | Selected LLM Judge: **{st.session_state.selected_llm}**")
        with col2:
            if st.button("🗑️ Remove", help="Remove uploaded file", key="delete_upload"):
                uploaded_file = None
                st.session_state.processing_complete = False
                st.session_state.dataset_processed = None
                st.rerun()
        
        # Simulate processing
        if st.button("🚀 Launch ADAMS Analysis", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                f"Initializing {st.session_state.selected_llm} evaluation matrix...",
                "Deploying multi-agent analysis swarm...",
                "Processing domain-specific parameters...",
                "Calibrating metric weighting algorithms...",
                "Synthesizing evaluation confidence scores...",
                "ADAMS processing complete ⚡"
            ]
            
            for i, stage in enumerate(stages):
                progress = (i + 1) / len(stages)
                progress_bar.progress(progress)
                status_text.markdown(f'<p class="neon-text">{stage}</p>', unsafe_allow_html=True)
                time.sleep(0.8)
            
            # Process the uploaded dataset
            processed_data = process_uploaded_dataset(uploaded_file, st.session_state.selected_llm)
            if processed_data:
                st.session_state.dataset_processed = processed_data
                st.session_state.processing_complete = True
                
                # Add to history for comparison
                dataset_entry = {
                    'name': f"{uploaded_file.name} ({st.session_state.selected_llm})",
                    'data': processed_data,
                    'llm_judge': st.session_state.selected_llm,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': uploaded_file.name,
                    'sample_count': len(processed_data)
                }
                st.session_state.processed_datasets_history.append(dataset_entry)
                
                st.success(f"✅ Successfully processed {len(processed_data)} samples with {st.session_state.selected_llm}!")
                time.sleep(1)
                st.session_state.page = 'dataset'
                st.rerun()
            else:
                st.error("❌ Failed to process dataset. Please check file format.")
    
    if not st.session_state.processing_complete:
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)
        st.info("✅ Processing complete! Check the **Dataset Review** page to see your results.")
        
        if st.button("📋 Go to Dataset Review", use_container_width=True, type="primary"):
            st.session_state.page = 'dataset'
            st.rerun()

# Page 2: Dataset Review
elif st.session_state.page == 'dataset':
    st.markdown("## 📋 Dataset Review & ADAMS Reconfiguration")
    st.markdown("Review the processed dataset with ADAMS scores and download results")
    
    if st.session_state.dataset_processed:
        # Add clear dataset option
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown(f"### 📊 Processed Dataset (LLM Judge: **{st.session_state.selected_llm}**)")
        with col2:
            st.markdown('<div style="padding-top: 2rem;">', unsafe_allow_html=True)
            if st.button("🗑️ Clear Dataset", help="Clear processed dataset and start over", type="secondary"):
                st.session_state.dataset_processed = None
                st.session_state.processing_complete = False
                st.session_state.page = 'upload'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display dataset as table
        df = pd.DataFrame(st.session_state.dataset_processed)
        
        # Show comparison between original and ADAMS-processed data
        st.markdown("#### 📊 ADAMS-Enhanced Dataset")
        st.markdown("*This dataset has been processed by ADAMS with additional evaluation metrics and scores.*")
        
        # Display the enhanced dataset
        display_df = df.copy()
        if 'Original_Data' in display_df.columns:
            display_df = display_df.drop('Original_Data', axis=1)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download options
        st.markdown("### 💾 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"adams_dataset_{st.session_state.selected_llm.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download as JSON
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"adams_dataset_{st.session_state.selected_llm.lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Statistics
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Average ADAMS Score", f"{df['ADAMS_Score'].mean():.2f}")
        with col3:
            st.metric("Highest Score", f"{df['ADAMS_Score'].max():.2f}")
        with col4:
            st.metric("Lowest Score", f"{df['ADAMS_Score'].min():.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🎛️ Proceed to Configuration", use_container_width=True, type="primary"):
            st.session_state.page = 'config'
            st.rerun()
    
    else:
        st.warning("⚠️ No processed dataset available. Please upload and process a dataset first.")
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.page = 'upload'
            st.rerun()

# Page 3: Configuration
elif st.session_state.page == 'config':
    st.markdown("## 🎛️ Metric Configuration")
    st.markdown("Real-time metric calibration with reviewer feedback system")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Metric Control Matrix")
        
        # Create sliders for each metric with real-time updates
        updated_weights = {}
        
        for metric_name, data in st.session_state.metrics_data.items():
            updated_weights[metric_name] = st.slider(
                f"**{metric_name}**",
                min_value=0.0,
                max_value=1.0,
                value=data['weight'],
                step=0.05,
                key=f"slider_{metric_name}",
                help=f"Current score: {data['score']}"
            )
        
        # Update the session state immediately when sliders change
        for metric_name in st.session_state.metrics_data:
            st.session_state.metrics_data[metric_name]['weight'] = updated_weights[metric_name]
        
        if st.button("↺ Reset to Defaults", use_container_width=True):
            st.session_state.metrics_data = default_metrics.copy()
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Reviewer Comments Section
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Reviewer Comments & Notes")
        
        # Text area for comments
        reviewer_comment = st.text_area(
            "Add your evaluation comments:",
            value=st.session_state.reviewer_comments.get('main_comment', ''),
            height=150,
            placeholder="Enter your thoughts on the evaluation criteria, weight adjustments, or overall assessment..."
        )
        
        # Save comment mode selection
        comment_mode = st.selectbox(
            "Save as mode:",
            ["Draft", "Review", "Final", "Custom"],
            help="Select the mode for saving your comments"
        )
        
        if comment_mode == "Custom":
            custom_mode = st.text_input("Enter custom mode name:")
            comment_mode = custom_mode if custom_mode else "Custom"
        
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("💾 Save Comments", use_container_width=True):
                st.session_state.reviewer_comments['main_comment'] = reviewer_comment
                st.session_state.reviewer_comments['mode'] = comment_mode
                st.session_state.reviewer_comments['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"Comments saved as '{comment_mode}' mode!")
        
        with col_load:
            if st.button("📋 Export Comments", use_container_width=True):
                if st.session_state.reviewer_comments:
                    comment_data = {
                        "comment": st.session_state.reviewer_comments.get('main_comment', ''),
                        "mode": st.session_state.reviewer_comments.get('mode', ''),
                        "timestamp": st.session_state.reviewer_comments.get('timestamp', ''),
                        "llm_judge": st.session_state.selected_llm,
                        "metric_weights": st.session_state.metrics_data
                    }
                    
                    st.download_button(
                        label="Download Comments",
                        data=json.dumps(comment_data, indent=2),
                        file_name=f"reviewer_comments_{comment_mode.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        
        # Calculate final score in real-time
        total_weighted_score = 0
        total_weights = 0
        
        for metric_name, data in st.session_state.metrics_data.items():
            total_weighted_score += data['score'] * data['weight']
            total_weights += data['weight']
        
        final_score = total_weighted_score / total_weights if total_weights > 0 else 0
        
        # Display final score with real-time updates
        st.markdown(f"""
        <div class="score-display">
            <h3 style="color: #b8bcc8; margin-bottom: 1rem;">Final ADAMS Score</h3>
            <div class="score-value">{final_score:.2f}</div>
            <div style="font-size: 0.9rem; color: #b8bcc8; margin-top: 0.5rem;">
                Based on {total_weights:.2f} total weight
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample analysis
        st.markdown("#### 📋 Sample Analysis")
        
        with st.expander("View Sample Q&A", expanded=True):
            st.markdown("""
            **Query:** What are the key benefits of using RAG systems in healthcare applications?
            
            **AI Response:** RAG systems in healthcare offer several key benefits: 1) Access to up-to-date medical research and guidelines, 2) Reduced hallucination through grounded responses, 3) Compliance with regulatory requirements through traceable sources, and 4) Personalized patient care through dynamic information retrieval.
            """)
        
        # Impact analysis
        sorted_metrics = sorted(st.session_state.metrics_data.items(), key=lambda x: x[1]['weight'], reverse=True)
        top_3_metrics = sorted_metrics[:3]
        top_names = [metric[0] for metric in top_3_metrics]
        avg_top_weight = sum(metric[1]['weight'] for metric in top_3_metrics) / 3
        
        impact_text = f"**🔥 Real-time Impact Analysis:**\n\n"
        impact_text += f"**Current emphasis:** {', '.join(top_names)}\n\n"
        
        if avg_top_weight > 0.8:
            impact_text += "**Status:** High-confidence configuration detected. System optimized for precision evaluation."
        elif avg_top_weight > 0.6:
            impact_text += "**Status:** Balanced configuration active. Moderate weighting across evaluation dimensions."
        else:
            impact_text += "**Status:** Low-weight configuration. Consider increasing key metric priorities for better accuracy."
        
        impact_text += f"\n\n**Total Active Weight:** {total_weights:.2f}"
        
        st.info(impact_text)
        
        # Current session info
        if st.session_state.reviewer_comments:
            st.markdown("#### 📝 Current Session")
            st.markdown(f"**Mode:** {st.session_state.reviewer_comments.get('mode', 'Not set')}")
            if st.session_state.reviewer_comments.get('timestamp'):
                st.markdown(f"**Last saved:** {st.session_state.reviewer_comments['timestamp']}")
        
        # Live metrics summary
        st.markdown("#### 📊 Live Metrics")
        highest_weight_metric = max(st.session_state.metrics_data.items(), key=lambda x: x[1]['weight'])
        lowest_weight_metric = min(st.session_state.metrics_data.items(), key=lambda x: x[1]['weight'])
        
        st.markdown(f"**Highest priority:** {highest_weight_metric[0]} ({highest_weight_metric[1]['weight']:.2f})")
        st.markdown(f"**Lowest priority:** {lowest_weight_metric[0]} ({lowest_weight_metric[1]['weight']:.2f})")
        st.markdown(f"**Current final score:** {final_score:.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Back to Dataset", use_container_width=True):
            st.session_state.page = 'dataset'
            st.rerun()
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True, type="primary"):
            config_data = {
                "metrics": st.session_state.metrics_data,
                "final_score": final_score,
                "llm_judge": st.session_state.selected_llm,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.download_button(
                label="Download Configuration",
                data=json.dumps(config_data, indent=2),
                file_name=f"adams_config_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    with col3:
        if st.button("📤 Export Full Report", use_container_width=True, type="primary"):
            full_report = {
                "dataset": st.session_state.dataset_processed,
                "metrics": st.session_state.metrics_data,
                "final_score": final_score,
                "llm_judge": st.session_state.selected_llm,
                "reviewer_comments": st.session_state.reviewer_comments,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.download_button(
                label="Download Full Report",
                data=json.dumps(full_report, indent=2),
                file_name=f"adams_full_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Page 4: Dataset Comparison
elif st.session_state.page == 'compare':
    st.markdown("## ⚖️ Dataset Comparison Analysis")
    st.markdown("Compare previously processed ADAMS datasets with advanced statistical analysis")
    
    # Check if there are processed datasets available
    if len(st.session_state.processed_datasets_history) < 2:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Ready to Compare Datasets?")
        
        if len(st.session_state.processed_datasets_history) == 0:
            st.warning("⚠️ **No processed datasets available.** You need to process at least 2 datasets to use comparison features.")
            st.markdown("""
            **Steps to get started:**
            1. Go to **Dataset Upload** page
            2. Upload and process your first dataset
            3. Upload and process a second dataset (you can use different LLM judges)
            4. Return here to compare them side by side
            """)
        else:
            st.warning("⚠️ **Only 1 dataset processed.** You need at least 2 datasets to compare.")
            st.markdown(f"""
            **Currently available:**
            • {st.session_state.processed_datasets_history[0]['name']}
            
            **To enable comparison:**
            1. Go to **Dataset Upload** page
            2. Process another dataset (try a different LLM judge!)
            3. Return here to compare performance
            """)
        
        if st.button("📊 Go to Dataset Upload", use_container_width=True, type="primary"):
            st.session_state.page = 'upload'
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Dataset Selection Interface
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown("### 📂 Select Datasets for Comparison")
        st.markdown(f"Choose 2 datasets from your {len(st.session_state.processed_datasets_history)} processed datasets")
        
        col1, col2 = st.columns(2)
        
        # Create dataset options
        dataset_options = []
        for i, dataset in enumerate(st.session_state.processed_datasets_history):
            option_label = f"{dataset['name']} ({dataset['sample_count']} samples)"
            dataset_options.append((i, option_label, dataset))
        
        with col1:
            st.markdown("#### 🔵 Dataset A")
            selected_a = st.selectbox(
                "Select first dataset:",
                options=[opt[0] for opt in dataset_options],
                format_func=lambda x: dataset_options[x][1],
                key="dataset_a_select"
            )
            
            if selected_a is not None:
                dataset_a = dataset_options[selected_a][2]
                st.session_state.comparison_selection['dataset_a'] = dataset_a
                
                # Show dataset info
                st.info(f"""
                **📊 Dataset Info:**
                • **LLM Judge:** {dataset_a['llm_judge']}
                • **Processed:** {dataset_a['timestamp']}
                • **Samples:** {dataset_a['sample_count']}
                • **Original File:** {dataset_a['filename']}
                """)
        
        with col2:
            st.markdown("#### 🔴 Dataset B")
            # Filter out the selected dataset A to prevent comparing dataset with itself
            available_b_options = [opt for opt in dataset_options if opt[0] != selected_a]
            
            if available_b_options:
                selected_b = st.selectbox(
                    "Select second dataset:",
                    options=[opt[0] for opt in available_b_options],
                    format_func=lambda x: next(opt[1] for opt in dataset_options if opt[0] == x),
                    key="dataset_b_select"
                )
                
                if selected_b is not None:
                    dataset_b = next(opt[2] for opt in dataset_options if opt[0] == selected_b)
                    st.session_state.comparison_selection['dataset_b'] = dataset_b
                    
                    # Show dataset info
                    st.info(f"""
                    **📊 Dataset Info:**
                    • **LLM Judge:** {dataset_b['llm_judge']}
                    • **Processed:** {dataset_b['timestamp']}
                    • **Samples:** {dataset_b['sample_count']}
                    • **Original File:** {dataset_b['filename']}
                    """)
            else:
                st.warning("No other datasets available for comparison.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show comparison analysis if both datasets are selected
        if (st.session_state.comparison_selection['dataset_a'] is not None and 
            st.session_state.comparison_selection['dataset_b'] is not None):
            
            dataset_a = st.session_state.comparison_selection['dataset_a']
            dataset_b = st.session_state.comparison_selection['dataset_b']
            df_a = pd.DataFrame(dataset_a['data'])
            df_b = pd.DataFrame(dataset_b['data'])
            name_a = dataset_a['name']
            name_b = dataset_b['name']
            
            # Statistical Overview
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Statistical Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-display">
                    <div class="metric-value">{len(df_a)}</div>
                    <div class="metric-name">{name_a.split('(')[0].strip()} Samples</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-display">
                    <div class="metric-value">{len(df_b)}</div>
                    <div class="metric-name">{name_b.split('(')[0].strip()} Samples</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                avg_diff = df_a['ADAMS_Score'].mean() - df_b['ADAMS_Score'].mean()
                color = "#00f5ff" if avg_diff >= 0 else "#ff006e"
                st.markdown(f"""
                <div class="metric-display">
                    <div class="metric-value" style="color: {color};">{avg_diff:+.2f}</div>
                    <div class="metric-name">Score Difference</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                # Calculate statistical significance
                try:
                    from scipy import stats
                    t_stat, p_value = stats.ttest_ind(df_a['ADAMS_Score'], df_b['ADAMS_Score'])
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    sig_color = "#00f5ff" if p_value < 0.05 else "#b8bcc8"
                    sig_display = f"p={p_value:.3f}"
                except:
                    significance = "N/A"
                    sig_color = "#b8bcc8"
                    sig_display = "N/A"
                
                st.markdown(f"""
                <div class="metric-display">
                    <div class="metric-value" style="color: {sig_color}; font-size: 1.5rem;">{significance}</div>
                    <div class="metric-name">{sig_display}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # LLM Judge Performance Comparison
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 🤖 LLM Judge Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### 🔵 {dataset_a['llm_judge']} Results")
                st.markdown(f"**Mean ADAMS Score:** {df_a['ADAMS_Score'].mean():.2f}")
                st.markdown(f"**Median Score:** {df_a['ADAMS_Score'].median():.2f}")
                st.markdown(f"**Standard Deviation:** {df_a['ADAMS_Score'].std():.2f}")
                st.markdown(f"**Score Range:** {df_a['ADAMS_Score'].min():.2f} - {df_a['ADAMS_Score'].max():.2f}")
                
                # Performance rating
                avg_score_a = df_a['ADAMS_Score'].mean()
                if avg_score_a >= 9.0:
                    rating_a = "🟢 Excellent"
                elif avg_score_a >= 8.0:
                    rating_a = "🔵 Very Good"
                elif avg_score_a >= 7.0:
                    rating_a = "🟡 Good"
                else:
                    rating_a = "🔴 Needs Improvement"
                
                st.markdown(f"**Performance Rating:** {rating_a}")
            
            with col2:
                st.markdown(f"#### 🔴 {dataset_b['llm_judge']} Results")
                st.markdown(f"**Mean ADAMS Score:** {df_b['ADAMS_Score'].mean():.2f}")
                st.markdown(f"**Median Score:** {df_b['ADAMS_Score'].median():.2f}")
                st.markdown(f"**Standard Deviation:** {df_b['ADAMS_Score'].std():.2f}")
                st.markdown(f"**Score Range:** {df_b['ADAMS_Score'].min():.2f} - {df_b['ADAMS_Score'].max():.2f}")
                
                # Performance rating
                avg_score_b = df_b['ADAMS_Score'].mean()
                if avg_score_b >= 9.0:
                    rating_b = "🟢 Excellent"
                elif avg_score_b >= 8.0:
                    rating_b = "🔵 Very Good"
                elif avg_score_b >= 7.0:
                    rating_b = "🟡 Good"
                else:
                    rating_b = "🔴 Needs Improvement"
                
                st.markdown(f"**Performance Rating:** {rating_b}")
            
            # Winner determination
            if avg_diff > 0.1:
                winner = f"🏆 **Winner: {dataset_a['llm_judge']}** (by {avg_diff:.2f} points)"
            elif avg_diff < -0.1:
                winner = f"🏆 **Winner: {dataset_b['llm_judge']}** (by {abs(avg_diff):.2f} points)"
            else:
                winner = "🤝 **Result: Statistical Tie** (difference < 0.1)"
            
            st.markdown(f"### {winner}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed Metric Comparison
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 🧬 Detailed Metric Analysis")
            
            # Get metric columns (excluding metadata)
            metric_cols = [col for col in df_a.columns if col not in ['Question', 'Reference_Answer', 'Model_Answer', 'LLM_Judge', 'Processing_Timestamp', 'Original_Data']]
            
            if len(metric_cols) > 1:
                comparison_data = []
                for metric in metric_cols:
                    if metric in df_b.columns:
                        try:
                            mean_a = df_a[metric].mean()
                            mean_b = df_b[metric].mean()
                            difference = mean_a - mean_b
                            
                            comparison_data.append({
                                'Metric': metric.replace('_', ' ').title(),
                                f'{dataset_a["llm_judge"]} Avg': round(mean_a, 2),
                                f'{dataset_b["llm_judge"]} Avg': round(mean_b, 2),
                                'Difference': round(difference, 2),
                                'Better Judge': dataset_a['llm_judge'] if difference > 0 else dataset_b['llm_judge'] if difference < 0 else 'Tie'
                            })
                        except:
                            continue
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Judge performance summary
                    a_wins = sum(1 for row in comparison_data if row['Better Judge'] == dataset_a['llm_judge'])
                    b_wins = sum(1 for row in comparison_data if row['Better Judge'] == dataset_b['llm_judge'])
                    ties = sum(1 for row in comparison_data if row['Better Judge'] == 'Tie')
                    
                    st.markdown(f"""
                    **📊 Metric Performance Summary:**
                    • **{dataset_a['llm_judge']}**: {a_wins} metrics won
                    • **{dataset_b['llm_judge']}**: {b_wins} metrics won  
                    • **Ties**: {ties} metrics
                    """)
                else:
                    st.info("No comparable metrics found between datasets.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Export Comparison
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 💾 Export Comparison Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create comparison report
                report_data = {
                    "comparison_summary": {
                        "dataset_a": {
                            "name": name_a,
                            "llm_judge": dataset_a['llm_judge'],
                            "samples": len(df_a),
                            "mean_score": df_a['ADAMS_Score'].mean(),
                            "processing_time": dataset_a['timestamp']
                        },
                        "dataset_b": {
                            "name": name_b,
                            "llm_judge": dataset_b['llm_judge'],
                            "samples": len(df_b),
                            "mean_score": df_b['ADAMS_Score'].mean(),
                            "processing_time": dataset_b['timestamp']
                        },
                        "comparison_results": {
                            "score_difference": avg_diff,
                            "statistical_significance": significance if 'significance' in locals() else "N/A",
                            "winner": dataset_a['llm_judge'] if avg_diff > 0.1 else dataset_b['llm_judge'] if avg_diff < -0.1 else "Tie",
                            "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    },
                    "detailed_metrics": comparison_data if 'comparison_data' in locals() else []
                }
                
                st.download_button(
                    label="📊 Download Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"adams_comparison_{dataset_a['llm_judge'].lower()}_vs_{dataset_b['llm_judge'].lower()}_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                if st.button("🔄 Change Selection", use_container_width=True):
                    st.session_state.comparison_selection = {'dataset_a': None, 'dataset_b': None}
                    st.rerun()
            
            with col3:
                if st.button("📊 New Analysis", use_container_width=True):
                    st.session_state.page = 'upload'
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            # Show helpful tips while user selects datasets
            st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
            st.markdown("### 💡 Comparison Tips")
            st.markdown("""
            **What you can compare:**
            - Different LLM judges on the same dataset
            - Same LLM judge on different datasets
            - Performance across different evaluation criteria
            - Statistical significance of differences
            
            **Best practices:**
            - Use datasets with similar question types for meaningful comparison
            - Consider sample size differences in your analysis
            - Look at both average scores and score distributions
            - Pay attention to statistical significance indicators
            """)
            st.markdown("</div>", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("### 🧠 ADAMS Interface")
    st.markdown("**Version:** 2.0.0")
    st.markdown("**Status:** ✅ Online")
    st.markdown("**Mode:** Interactive Demo")
    
    st.markdown("---")
    st.markdown("### 📊 Current Session")
    st.markdown(f"**Metrics Loaded:** {len(st.session_state.metrics_data)}")
    st.markdown(f"**Page:** {st.session_state.page.title()}")
    st.markdown(f"**LLM Judge:** {st.session_state.selected_llm}")
    
    if st.session_state.processing_complete:
        # Recalculate final score for sidebar display
        weighted_sum = sum(data['score'] * data['weight'] for data in st.session_state.metrics_data.values())
        total_weight = sum(data['weight'] for data in st.session_state.metrics_data.values())
        current_final_score = weighted_sum / total_weight if total_weight > 0 else 0
        st.markdown(f"**Current Score:** {current_final_score:.2f}")
    
    # Show comparison status
    if st.session_state.page == 'compare':
        st.markdown("---")
        st.markdown("### ⚖️ Available Datasets")
        if len(st.session_state.processed_datasets_history) == 0:
            st.markdown("❌ **No datasets processed**")
        else:
            st.markdown(f"✅ **{len(st.session_state.processed_datasets_history)} datasets ready**")
            for i, dataset in enumerate(st.session_state.processed_datasets_history, 1):
                st.markdown(f"{i}. {dataset['llm_judge']} ({dataset['sample_count']} samples)")
        
        if (st.session_state.comparison_selection['dataset_a'] is not None and 
            st.session_state.comparison_selection['dataset_b'] is not None):
            st.markdown("🎯 **Comparison Active**")
    
    # Show weight distribution
    if st.session_state.metrics_data:
        st.markdown("---")
        st.markdown("### ⚖️ Weight Summary")
        total_weight = sum(data['weight'] for data in st.session_state.metrics_data.values())
        st.markdown(f"**Total Weight:** {total_weight:.2f}")
        
        # Show top 3 metrics
        top_metrics = sorted(st.session_state.metrics_data.items(), key=lambda x: x[1]['weight'], reverse=True)[:3]
        st.markdown("**Top 3 Priorities:**")
        for i, (name, data) in enumerate(top_metrics, 1):
            st.markdown(f"{i}. {name}: {data['weight']:.2f}")
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.metrics_data = default_metrics.copy()
        st.session_state.processing_complete = False
        st.session_state.dataset_processed = None
        st.session_state.reviewer_comments = {}
        st.session_state.page = 'upload'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 Demo Instructions")
    st.markdown("""
    1. **Upload:** Select file and LLM judge
    2. **Dataset:** Review processed results
    3. **Configuration:** Adjust weights & add comments
    4. **Export:** Download configurations and reports
    """)
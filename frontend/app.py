import streamlit as st
import time
import json
import pandas as pd
import random
from datetime import datetime
from backend_client import get_backend_client

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
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, 
                                    #16213e 100%);
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
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'error_generation'
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = 'Qwen'
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


def clean_dataset(df):
    """Clean dataset by removing NaN values and empty rows"""
    # Remove rows where all values are NaN
    df_cleaned = df.dropna(how='all')
    
    # Keep only required columns if they exist
    required_columns = ['question', 'response', 'documents']
    existing_columns = [col for col in required_columns if col in df_cleaned.columns]
    
    if len(existing_columns) < 3:
        # Try to infer column names
        cols = df_cleaned.columns.tolist()
        if len(cols) >= 3:
            # Use first 3 columns as required columns
            df_cleaned = df_cleaned.iloc[:, :3]
            df_cleaned.columns = required_columns
        else:
            return None, "Dataset must have at least 3 columns: question, response, documents"
    else:
        df_cleaned = df_cleaned[existing_columns]
    
    # Remove rows with NaN in required columns
    df_cleaned = df_cleaned.dropna(subset=required_columns)
    
    # Replace any remaining NaN values with empty strings
    df_cleaned = df_cleaned.fillna('')
    
    return df_cleaned, f"Cleaned dataset: {len(df_cleaned)} valid samples"

def generate_synthetic_errors_with_backend(df, error_types):
    """Use backend API for synthetic error generation"""
    try:
        # Clean dataset first
        df_cleaned, message = clean_dataset(df)
        if df_cleaned is None:
            st.error(f"Dataset cleaning failed: {message}")
            return None
        
        st.info(message)
        
        # Convert dataframe to format expected by backend
        dataset = df_cleaned.to_dict('records')
        
        # Process data annotation using original pipeline (backend will validate HuggingFace format)
        with st.spinner("Running data annotation pipeline with LLM..."):
            result = st.session_state.backend_client.start_data_annotation(dataset)
            
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
        # Fallback to simple generation if backend fails
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


def calculate_evaluation_scores_with_backend(df, llm_judge, metrics_weights):
    """Use backend API for evaluation"""
    try:
        # Clean dataset first
        df_cleaned, message = clean_dataset(df)
        if df_cleaned is None:
            st.error(f"Dataset cleaning failed: {message}")
            return None
        
        # Convert dataframe to format expected by backend
        dataset = df_cleaned.to_dict('records')
        
        # Get available evaluators from backend
        available_evaluators = st.session_state.backend_client.get_available_evaluators()
        evaluator_names = [ev['name'] for ev in available_evaluators]
        
        # Convert metrics_weights to evaluator weights
        evaluator_weights = {}
        for metric_name, data in metrics_weights.items():
            # Map frontend metric names to backend evaluator names
            for ev_name in evaluator_names:
                if metric_name.lower().replace(' ', '') in ev_name.lower():
                    evaluator_weights[ev_name] = data['weight']
                    break
        
        # Start evaluation task
        task_id = st.session_state.backend_client.start_evaluation(
            dataset=dataset,
            selected_evaluators=list(evaluator_weights.keys()),
            metric_weights=evaluator_weights,
            llm_provider="openai",
            model_name="gpt-4o-mini"
        )
        
        if task_id:
            # Poll for results with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                try:
                    progress = st.session_state.backend_client.get_evaluation_progress(task_id)
                    
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
                        # Update progress
                        prog_info = progress.get('progress', {})
                        if isinstance(prog_info, dict):
                            pct = prog_info.get('progress_percentage', 0)
                            progress_bar.progress(pct / 100)
                            status_text.text(f"Evaluating: {prog_info.get('current_evaluator', 'Processing...')}")
                        else:
                            progress_bar.progress(0.3)
                            status_text.text("Evaluation in progress...")
                
                except Exception as e:
                    st.error(f"Error checking progress: {str(e)}")
                    break
                
                time.sleep(3)  # Check every 3 seconds
            
            st.error("Timeout: Evaluation took too long")
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
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #b8bcc8; margin-bottom: 2rem;">Adaptive Domain-Aware Metric Selection</p>', unsafe_allow_html=True)

# Backend status indicator
if backend_connected:
    st.success("🟢 Backend API Connected")
else:
    st.warning("🟡 Backend API Unavailable - Using Fallback Mode")

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

# Tab 1: Error Generation
if st.session_state.current_tab == 'error_generation':
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("## 🔧 Data Annotation Pipeline")
    st.markdown("Upload dataset and run the original annotation pipeline (Key Points, Mistake Distribution, Answer Generation)")
    
    # Sample dataset download
    st.markdown("### 📥 Sample Dataset")
    sample_csv_data = """Question,Reference_Answer,Model_Answer
"What are the key benefits of using RAG systems in healthcare applications?","RAG systems provide up-to-date medical information, reduce hallucinations, ensure compliance, and enable personalized care.","RAG systems in healthcare offer several key benefits: 1) Access to up-to-date medical research and guidelines, 2) Reduced hallucination through grounded responses, 3) Compliance with regulatory requirements through traceable sources, and 4) Personalized patient care through dynamic information retrieval."
"How do transformer architectures handle long sequences?","Transformers use attention mechanisms but face quadratic complexity with sequence length, leading to various optimization techniques.","Transformer architectures handle long sequences through self-attention mechanisms, though they face computational challenges due to quadratic complexity. Modern approaches include attention optimization, sparse attention patterns, and hierarchical processing to manage memory and computational requirements effectively."
"What is the difference between supervised and unsupervised learning?","Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data.","Supervised learning algorithms learn from labeled training data to make predictions on new data, while unsupervised learning discovers hidden patterns and structures in data without labels, such as clustering and dimensionality reduction techniques."
"Explain the concept of transfer learning in deep learning.","Transfer learning involves using pre-trained models and adapting them to new tasks, reducing training time and data requirements.","Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a related task. This approach leverages knowledge gained from pre-trained models, significantly reducing training time and computational resources while often achieving better performance."
"What are the main challenges in implementing RAG systems?","Key challenges include retrieval quality, context length limitations, computational costs, and maintaining consistency between retrieved and generated content.","The main challenges in implementing RAG systems include: 1) Ensuring high-quality and relevant document retrieval, 2) Managing context length limitations in language models, 3) Balancing computational costs with performance, 4) Maintaining consistency between retrieved information and generated responses, and 5) Handling conflicting information from multiple sources." """
    
    st.download_button(
        label="📥 Download Sample Dataset (CSV)",
        data=sample_csv_data,
        file_name="sample_dataset.csv",
        mime="text/csv"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload original dataset",
        type=['csv', 'json'],
        help="Support CSV, JSON formats"
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
            
        except Exception as e:
            st.error(f"❌ File loading failed: {str(e)}")
    
    # Error type selection
    if st.session_state.original_dataset is not None:
        st.markdown("### 🎯 Error Type Configuration")
        
        error_options = {
            'Entity_Error': 'Entity Error - Replace key entity information',
            'Negation': 'Negation Error - Change semantic polarity',
            'Missing_Information': 'Missing Information - Truncate or omit key information',
            'Out_of_Reference': 'Out of Reference - Add irrelevant or incorrect information',
            'Numerical_Error': 'Numerical Error - Modify numbers or statistical information'
        }
        
        selected_errors = []
        for error_type, description in error_options.items():
            if st.checkbox(description, key=f"error_{error_type}"):
                selected_errors.append(error_type)
        
        # Generate error dataset
        if selected_errors and st.button("🚀 Generate Error Dataset", use_container_width=True, type="primary"):
            if backend_connected:
                error_df = generate_synthetic_errors_with_backend(st.session_state.original_dataset, selected_errors)
            else:
                with st.spinner("Generating synthetic errors (fallback mode)..."):
                    error_df = generate_synthetic_errors_fallback(st.session_state.original_dataset, selected_errors)
            
            if error_df is not None:
                st.session_state.error_dataset = error_df
                
                st.success(f"✅ Successfully generated {len(error_df)} samples with errors")
                
                # Show error statistics
                error_counts = error_df['Error_Type'].value_counts()
                st.markdown("#### 📊 Error Type Distribution")
                for error_type, count in error_counts.items():
                    st.markdown(f"- **{error_options[error_type]}**: {count} samples")
    
    # Display generated error dataset
    if st.session_state.error_dataset is not None:
        st.markdown("### 📋 Generated Error Dataset")
        st.dataframe(st.session_state.error_dataset, use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.error_dataset.to_csv(index=False)
            st.download_button(
                label="📥 Download Error Dataset (CSV)",
                data=csv_data,
                file_name=f"error_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = st.session_state.error_dataset.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download Error Dataset (JSON)",
                data=json_data,
                file_name=f"error_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Evaluation
elif st.session_state.current_tab == 'evaluation':
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Evaluation & Weight Adjustment")
    st.markdown("Evaluate datasets and adjust evaluation metric weights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset selection
        st.markdown("### 📂 Select Dataset for Evaluation")
        
        dataset_options = []
        if st.session_state.original_dataset is not None:
            dataset_options.append("Original Dataset")
        if st.session_state.error_dataset is not None:
            dataset_options.append("Error Dataset")
        
        if dataset_options:
            selected_dataset = st.selectbox("Select dataset to evaluate:", dataset_options)
            
            if selected_dataset == "Original Dataset":
                current_dataset = st.session_state.original_dataset
            else:
                current_dataset = st.session_state.error_dataset
        else:
            st.warning("⚠️ Please upload dataset in Data Annotation page first")
            current_dataset = None
        
        # LLM Judge selection
        st.markdown("### 🤖 Select LLM Evaluator")
        llm_options = ['Qwen', 'Deepseek', 'Distilled Qwen', 
                       'Mistral', 'LLaMA 3.1']
        st.session_state.selected_llm = st.selectbox(
            "Select LLM evaluator:",
            llm_options,
            index=llm_options.index(st.session_state.selected_llm)
        )
        
        # Metric weight adjustment
        st.markdown("### ⚖️ Evaluation Metric Weight Configuration")
        
        updated_weights = {}
        for metric_name, data in st.session_state.metrics_data.items():
            updated_weights[metric_name] = st.slider(
                f"**{metric_name}**",
                min_value=0.0,
                max_value=1.0,
                value=data['weight'],
                step=0.05,
                key=f"eval_slider_{metric_name}",
                help=f"Current score: {data['score']}"
            )
        
        # Update weights
        for metric_name in st.session_state.metrics_data:
            st.session_state.metrics_data[metric_name]['weight'] = updated_weights[metric_name]
        
        if st.button("↺ Reset to Default Weights", use_container_width=True):
            st.session_state.metrics_data = default_metrics.copy()
            st.rerun()
        
        # Start evaluation
        if current_dataset is not None and st.button("🚀 Start Evaluation", use_container_width=True, type="primary"):
            with st.spinner(f"Evaluating with {st.session_state.selected_llm}..."):
                progress_bar = st.progress(0)
                for i in range(5):
                    time.sleep(0.5)
                    progress_bar.progress((i + 1) / 5)
                
                # Calculate evaluation scores
                if backend_connected:
                    evaluated_df = calculate_evaluation_scores_with_backend(
                        current_dataset, 
                        st.session_state.selected_llm, 
                        st.session_state.metrics_data
                    )
                else:
                    evaluated_df = calculate_evaluation_scores_fallback(
                        current_dataset, 
                        st.session_state.selected_llm, 
                        st.session_state.metrics_data
                    )
                
                # Save to history
                evaluation_record = {
                    'id': len(st.session_state.evaluation_history) + 1,
                    'name': f"{selected_dataset} - {st.session_state.selected_llm}",
                    'dataset_type': selected_dataset,
                    'llm_judge': st.session_state.selected_llm,
                    'data': evaluated_df,
                    'metrics_config': st.session_state.metrics_data.copy(),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'sample_count': len(evaluated_df)
                }
                
                st.session_state.evaluation_history.append(evaluation_record)
                st.success(f"✅ Evaluation completed! Evaluated {len(evaluated_df)} records")
    
    with col2:
        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        
        # Real-time weighted score calculation
        total_weighted_score = 0
        total_weights = 0
        
        for metric_name, data in st.session_state.metrics_data.items():
            total_weighted_score += data['score'] * data['weight']
            total_weights += data['weight']
        
        final_score = total_weighted_score / total_weights if total_weights > 0 else 0
        
        # Display current configuration score
        st.markdown(f"""
        <div class="score-display">
            <h3 style="color: #b8bcc8; margin-bottom: 1rem;">Current Configuration Score</h3>
            <div class="score-value">{final_score:.2f}</div>
            <div style="font-size: 0.9rem; color: #b8bcc8; margin-top: 0.5rem;">
                Total Weights: {total_weights:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Weight distribution display
        st.markdown("#### ⚖️ Weight Distribution")
        sorted_metrics = sorted(st.session_state.metrics_data.items(), 
                               key=lambda x: x[1]['weight'], reverse=True)
        
        for i, (metric_name, data) in enumerate(sorted_metrics[:5], 1):
            st.markdown(f"{i}. **{metric_name}**: {data['weight']:.2f}")
        
        # Evaluation history summary
        st.markdown("#### 📈 Evaluation History")
        if st.session_state.evaluation_history:
            st.markdown(f"Completed evaluations: **{len(st.session_state.evaluation_history)}** times")
            latest = st.session_state.evaluation_history[-1]
            st.markdown(f"Latest evaluation: {latest['name']}")
            st.markdown(f"Time: {latest['timestamp']}")
        else:
            st.markdown("No evaluation history yet")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
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
            avg_score = record['data']['ADAMS_Score'].mean()
            history_data.append({
                'ID': record['id'],
                'Name': record['name'],
                'Dataset Type': record['dataset_type'],
                'LLM Judge': record['llm_judge'],
                'Sample Count': record['sample_count'],
                'Average Score': f"{avg_score:.2f}",
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
                
                avg_a = df_a['ADAMS_Score'].mean()
                avg_b = df_b['ADAMS_Score'].mean()
                score_diff = avg_a - avg_b
                
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
    st.markdown("**Status**: ✅ Online")
    
    st.markdown("---")
    st.markdown("### 📊 Current Session")
    st.markdown(f"**Current Tab**: {st.session_state.current_tab}")
    st.markdown(f"**LLM Judge**: {st.session_state.selected_llm}")
    
    # Dataset status
    st.markdown("### 📂 Dataset Status")
    if st.session_state.original_dataset is not None:
        st.markdown(f"✅ **Original Dataset**: {len(st.session_state.original_dataset)} samples")
    else:
        st.markdown("❌ **Original Dataset**: Not loaded")
    
    if st.session_state.error_dataset is not None:
        st.markdown(f"✅ **Error Dataset**: {len(st.session_state.error_dataset)} samples")
    else:
        st.markdown("❌ **Error Dataset**: Not generated")
    
    # Evaluation history
    st.markdown("### 📈 Evaluation History")
    st.markdown(f"**History Count**: {len(st.session_state.evaluation_history)}")
    
    if st.session_state.evaluation_history:
        latest = st.session_state.evaluation_history[-1]
        st.markdown(f"**Latest Evaluation**: {latest['name'][:20]}...")
        avg_score = latest['data']['ADAMS_Score'].mean()
        st.markdown(f"**Average Score**: {avg_score:.2f}")
    
    # Quick actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in ['original_dataset', 'error_dataset', 'evaluation_history', 'comparison_selection']:
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
    1. **Error Generation**: Upload dataset, generate synthetic errors
    2. **Evaluation**: Configure weights, calculate evaluation scores
    3. **History**: Compare different evaluation results
    
    **Tips**: 
    - Support multiple error type combinations
    - Weights affect final scores in real-time
    - History records can export analysis reports
    """)
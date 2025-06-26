import streamlit as st
import pandas as pd
import time
import json
import random
from backend_client import (
    test_backend_connection,
    start_data_augmentation,
    get_augmentation_result,
    poll_augmentation_progress,
    validate_hf_dataset,
    process_dataset_with_backend,
    get_available_evaluators_from_backend
)

st.set_page_config(
    page_title="RAG Evaluation Two-Step Workflow", 
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'original_dataset' not in st.session_state:
    st.session_state.original_dataset = None
if 'augmented_dataset' not in st.session_state:
    st.session_state.augmented_dataset = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = 'OpenAI'

st.title("🔬 RAG Evaluation Two-Step Workflow")
st.markdown("**Step 1:** Upload HF dataset → Generate answers with mistakes")
st.markdown("**Step 2:** Evaluate augmented dataset → Adjust metric weights")

with st.sidebar:
    st.header("Workflow Progress")
    
    if st.session_state.step == 1:
        st.success("✅ Step 1: Data Augmentation")
        st.info("⏳ Step 2: Evaluation")
    else:
        st.success("✅ Step 1: Data Augmentation")
        st.success("✅ Step 2: Evaluation")
    
    st.divider()
    
    st.header("LLM Configuration")
    llm_options = ['OpenAI', 'Qwen', 'DeepSeek', 'Mistral', 'Local']
    st.session_state.selected_llm = st.selectbox(
        "Select LLM Judge:",
        llm_options,
        index=llm_options.index(st.session_state.selected_llm)
    )
    
    st.divider()
    
    if st.button("🔄 Reset Workflow", type="secondary"):
        st.session_state.step = 1
        st.session_state.original_dataset = None
        st.session_state.augmented_dataset = None
        st.session_state.evaluation_results = None
        st.rerun()

# Sidebar step navigation
st.sidebar.title("Two-Step Workflow")
current_step = st.sidebar.radio(
    "选择步骤:",
    ["Step 1: 数据增强", "Step 2: 评估分析"],
    index=0 if st.session_state.step == 1 else 1
)

# Update step based on sidebar selection
if current_step == "Step 1: 数据增强":
    st.session_state.step = 1
elif current_step == "Step 2: 评估分析":
    st.session_state.step = 2

if st.session_state.step == 1:
    st.header("Step 1: Data Augmentation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Hugging Face Dataset")
        uploaded_file = st.file_uploader(
            "Upload dataset (CSV/JSON/Parquet)",
            type=['csv', 'json', 'parquet'],
            help="Dataset must contain: question, response, documents columns"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                
                st.success(f"Dataset loaded: {len(df)} samples")
                st.dataframe(df.head(3), use_container_width=True)
                
                st.session_state.original_dataset = df
                
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    with col2:
        st.subheader("Augmentation Settings")
        
        mistake_types = st.multiselect(
            "Mistake Types:",
            ['Entity_Error', 'Negation', 'Missing_Information', 'Out_of_Reference', 'Numerical_Error'],
            default=['Entity_Error', 'Negation', 'Missing_Information']
        )
        
        num_mistakes = st.slider("Mistakes per Answer:", 1, 5, 3)
        
        if st.session_state.original_dataset is not None:
            if st.button("🚀 Start Data Augmentation", type="primary", use_container_width=True):
                
                dataset_dict = st.session_state.original_dataset.to_dict('records')
                
                if test_backend_connection():
                    result = start_data_augmentation(
                        dataset=dataset_dict,
                        llm_provider=st.session_state.selected_llm.lower(),
                        mistake_types=mistake_types,
                        num_mistakes=num_mistakes
                    )
                    
                    if result:
                        task_id = result['task_id']
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("Processing data augmentation..."):
                            def update_progress(progress_data):
                                if progress_data:
                                    progress = progress_data.get('progress', 0)
                                    message = progress_data.get('message', 'Processing...')
                                    progress_bar.progress(progress)
                                    status_text.text(message)
                            
                            final_progress = poll_augmentation_progress(task_id, update_progress)
                            
                            if final_progress and final_progress.get('status') == 'completed':
                                augmentation_result = get_augmentation_result(task_id)
                                
                                if augmentation_result:
                                    st.session_state.augmented_dataset = pd.DataFrame(
                                        augmentation_result['augmented_dataset']
                                    )
                                    st.success("Data augmentation completed!")
                                    st.session_state.step = 2
                                    st.rerun()
                            else:
                                st.error("Data augmentation failed")
                    else:
                        st.error("Failed to start data augmentation")
                else:
                    st.warning("Backend not available, using simulated augmentation...")
                    
                    augmented_data = []
                    for _, row in st.session_state.original_dataset.iterrows():
                        
                        mistakes = ['entity error', 'negation error', 'missing info']
                        generated_answer = f"{row.get('response', 'Default response')} [Simulated with {', '.join(mistakes)}]"
                        
                        augmented_row = {
                            'Question': row.get('question', ''),
                            'Reference_Answer': row.get('response', ''),
                            'Model_Answer': generated_answer,
                            'Context': row.get('documents', ''),
                            'Mistake_Types': mistakes,
                            'Num_Mistakes': len(mistakes)
                        }
                        augmented_data.append(augmented_row)
                    
                    st.session_state.augmented_dataset = pd.DataFrame(augmented_data)
                    st.success("Simulated data augmentation completed!")
                    st.session_state.step = 2
                    st.rerun()

elif st.session_state.step == 2:
    st.header("Step 2: Evaluation & Metric Adjustment")
    
    # Check if we have data, if not allow direct upload
    if st.session_state.augmented_dataset is None:
        st.info("💡 您可以直接上传包含生成答案的数据集进行评估，无需完成Step 1")
        
        # File upload for direct evaluation
        uploaded_file = st.file_uploader(
            "上传评估数据集",
            type=['csv', 'json', 'parquet'],
            help="数据集应包含：Question, Reference_Answer, Model_Answer, Context（可选）"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                
                # Validate required columns
                required_cols = ['Question', 'Reference_Answer', 'Model_Answer']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ 缺少必需列: {missing_cols}")
                    st.info("💡 请确保数据集包含以下列：Question, Reference_Answer, Model_Answer")
                else:
                    st.session_state.augmented_dataset = df
                    st.success(f"✅ 成功加载 {len(df)} 条评估数据")
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ 文件加载失败: {str(e)}")
        
        # Show sample format
        with st.expander("📋 查看期望的数据格式"):
            sample_data = pd.DataFrame({
                'Question': ['什么是机器学习？', '深度学习的优势是什么？'],
                'Reference_Answer': ['机器学习是人工智能的子领域...', '深度学习具有强大的特征学习能力...'],
                'Model_Answer': ['机器学习是让计算机自动学习的技术...', '深度学习可以自动提取特征...'],
                'Context': ['相关背景知识...', '技术背景信息...']
            })
            st.dataframe(sample_data, use_container_width=True)
    
    # Main evaluation interface (same as before but with dynamic metric pool)
    if st.session_state.augmented_dataset is not None:
        
        tab1, tab2, tab3 = st.tabs(["📊 Evaluation", "⚖️ Metric Weights", "📥 Download"])
        
        with tab1:
            st.subheader("Dataset for Evaluation")
            st.dataframe(st.session_state.augmented_dataset, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔬 Start Evaluation", type="primary", use_container_width=True):
                    
                    evaluation_dataset = st.session_state.augmented_dataset.to_dict('records')
                    
                    if test_backend_connection():
                        # 使用后端获取可用的评估器
                        available_evaluators = get_available_evaluators_from_backend()
                        
                        # 让用户选择要使用的评估器
                        if available_evaluators:
                            with st.expander("🔧 选择评估指标", expanded=True):
                                selected_evaluators = []
                                for evaluator in available_evaluators:
                                    if st.checkbox(
                                        f"{evaluator.get('name', 'Unknown')}",
                                        value=True,
                                        help=evaluator.get('description', 'No description available')
                                    ):
                                        selected_evaluators.append(evaluator.get('class_name'))
                        else:
                            selected_evaluators = None
                        
                        if st.button("确认并开始评估"):
                            result = process_dataset_with_backend(
                                dataset=evaluation_dataset,
                                llm_provider=st.session_state.selected_llm.lower(),
                                selected_evaluators=selected_evaluators
                            )
                            
                            if result:
                                st.session_state.evaluation_results = result
                                st.session_state.available_metrics = [metric['name'] for metric in result.get('metrics', [])]
                                st.success("Evaluation completed!")
                                st.rerun()
                    else:
                        st.warning("Backend not available, using simulated evaluation...")
                        
                        # 模拟评估器池 - 从后端evaluator模块获取
                        simulated_metrics = [
                            'Factual_Accuracy', 'Coherence', 'Context_Relevance', 
                            'Context_Utilization', 'Faithfulness', 'Answer_Similarity'
                        ]
                        
                        simulated_results = []
                        for _, row in st.session_state.augmented_dataset.iterrows():
                            result_row = row.to_dict()
                            
                            # 为每个指标生成随机分数
                            for metric in simulated_metrics:
                                result_row[metric] = random.uniform(0.6, 0.95)
                            
                            # 计算总分
                            result_row['ADAMS_Score'] = sum(result_row[metric] for metric in simulated_metrics) / len(simulated_metrics)
                            simulated_results.append(result_row)
                        
                        st.session_state.evaluation_results = simulated_results
                        st.session_state.available_metrics = simulated_metrics
                        st.success("Simulated evaluation completed!")
                        st.rerun()
            
            with col2:
                if st.button("← Back to Step 1", type="secondary", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
        
        with tab2:
            if st.session_state.evaluation_results:
                st.subheader("Adjust Metric Weights")
                
                results_df = pd.DataFrame(st.session_state.evaluation_results)
                available_metrics = st.session_state.get('available_metrics', ['Factual_Accuracy', 'Coherence', 'Relevance', 'Completeness'])
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Dynamic Metric Weights:**")
                    
                    weights = {}
                    # 动态生成权重滑块，基于实际可用的指标
                    for metric in available_metrics:
                        if metric in results_df.columns:
                            default_weight = 0.8 if 'accuracy' in metric.lower() or 'factual' in metric.lower() else 0.7
                            weights[metric] = st.slider(
                                metric.replace('_', ' ').title(), 
                                0.0, 1.0, default_weight, 0.1,
                                key=f"weight_{metric}"
                            )
                    
                    if st.button("🔄 Recalculate Scores"):
                        
                        for idx, row in results_df.iterrows():
                            weighted_score = 0
                            total_weight = 0
                            
                            for metric, weight in weights.items():
                                if metric in row and pd.notna(row[metric]):
                                    weighted_score += row[metric] * weight
                                    total_weight += weight
                            
                            if total_weight > 0:
                                final_score = weighted_score / total_weight
                                results_df.at[idx, 'Weighted_Score'] = round(final_score, 3)
                        
                        st.session_state.evaluation_results = results_df.to_dict('records')
                        st.success("Scores recalculated with dynamic metrics!")
                        st.rerun()
                
                with col2:
                    st.write("**Evaluation Results:**")
                    
                    display_cols = ['Question']
                    if 'ADAMS_Score' in results_df.columns:
                        display_cols.append('ADAMS_Score')
                    if 'Weighted_Score' in results_df.columns:
                        display_cols.append('Weighted_Score')
                    
                    # 添加前几个指标列用于显示
                    metric_cols = [col for col in available_metrics[:3] if col in results_df.columns]
                    display_cols.extend(metric_cols)
                    
                    st.dataframe(
                        results_df[display_cols], 
                        use_container_width=True
                    )
                    
                    # 显示平均分数
                    col_a, col_b = st.columns(2)
                    with col_a:
                        avg_score = results_df['ADAMS_Score'].mean() if 'ADAMS_Score' in results_df.columns else 0
                        st.metric("Original Avg Score", f"{avg_score:.3f}")
                    with col_b:
                        weighted_avg = results_df['Weighted_Score'].mean() if 'Weighted_Score' in results_df.columns else avg_score
                        st.metric("Weighted Avg Score", f"{weighted_avg:.3f}")
        
        with tab3:
            st.subheader("Download Results")
            
            if st.session_state.evaluation_results:
                results_df = pd.DataFrame(st.session_state.evaluation_results)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_data,
                        file_name=f"evaluation_results_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    json_data = results_df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_data,
                        file_name=f"evaluation_results_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    if st.session_state.augmented_dataset is not None:
                        augmented_csv = st.session_state.augmented_dataset.to_csv(index=False)
                        st.download_button(
                            "📥 Download Augmented Dataset",
                            data=augmented_csv,
                            file_name=f"augmented_dataset_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    else:
        st.warning("No augmented dataset available. Please complete Step 1 first.")
        if st.button("← Go to Step 1"):
            st.session_state.step = 1
            st.rerun()

st.divider()
st.caption("RAG Evaluation Framework - Two-Step Workflow") 
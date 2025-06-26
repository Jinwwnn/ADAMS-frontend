import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import backend_client

st.set_page_config(
    page_title="RAG评估结果对比",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 RAG评估结果对比")
    
    # Check backend connection
    if not backend_client.test_connection():
        st.error("❌ 无法连接到后端服务。请确保后端服务正在运行。")
        st.stop()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "选择页面",
        ["评估历史", "结果对比", "新建评估"]
    )
    
    if page == "评估历史":
        show_evaluation_history()
    elif page == "结果对比":
        show_comparison_page()
    elif page == "新建评估":
        show_new_evaluation()

def show_evaluation_history():
    st.header("📋 评估历史")
    
    # Get evaluation history
    history_response = backend_client.get_evaluation_history()
    if not history_response:
        st.warning("无法获取评估历史")
        return
    
    history = history_response.get('history', [])
    if not history:
        st.info("暂无评估历史记录")
        return
    
    # Convert to DataFrame for display
    df_history = pd.DataFrame(history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Display history table
    st.subheader(f"共 {len(history)} 条记录")
    
    for i, record in enumerate(history):
        with st.expander(f"📊 {record['name']} - {record['timestamp'][:16]}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**LLM提供商**: {record['llm_provider']}")
                if record.get('model_name'):
                    st.write(f"**模型**: {record['model_name']}")
                
                # Dataset info
                dataset_info = record.get('dataset_info', {})
                st.write(f"**数据集大小**: {dataset_info.get('size', 'N/A')}")
                
                # Results summary
                results = record.get('results_summary', {})
                if 'final_score' in results:
                    st.write(f"**总体得分**: {results['final_score']:.3f}")
                
                sample_count = results.get('sample_count', record.get('dataset_info', {}).get('size', 'N/A'))
                st.write(f"**样本数量**: {sample_count}")
                
                # Notes
                if record.get('notes'):
                    st.write(f"**备注**: {record['notes']}")
            
            with col2:
                # Edit notes
                new_notes = st.text_area(
                    "编辑备注",
                    value=record.get('notes', ''),
                    key=f"notes_{record['id']}"
                )
                
                if st.button("保存备注", key=f"save_{record['id']}"):
                    result = backend_client.update_evaluation_notes(record['id'], new_notes)
                    if result:
                        st.success("备注已更新")
                        st.rerun()
                    else:
                        st.error("更新失败")
                
                if st.button("删除", key=f"delete_{record['id']}", type="secondary"):
                    result = backend_client.delete_evaluation(record['id'])
                    if result:
                        st.success("记录已删除")
                        st.rerun()
                    else:
                        st.error("删除失败")

def show_comparison_page():
    st.header("🔍 结果对比")
    
    # Get evaluation history
    history_response = backend_client.get_evaluation_history()
    if not history_response:
        st.warning("无法获取评估历史")
        return
    
    history = history_response.get('history', [])
    if len(history) < 2:
        st.info("至少需要2条评估记录才能进行对比")
        return
    
    # Selection for comparison
    st.subheader("选择要对比的评估结果")
    
    evaluation_options = {
        f"{record['name']} ({record['timestamp'][:16]})": record['id'] 
        for record in history
    }
    
    selected_evaluations = st.multiselect(
        "选择评估结果（至少选择2个）",
        options=list(evaluation_options.keys()),
        default=list(evaluation_options.keys())[:2] if len(evaluation_options) >= 2 else []
    )
    
    if len(selected_evaluations) < 2:
        st.warning("请至少选择2个评估结果进行对比")
        return
    
    selected_ids = [evaluation_options[name] for name in selected_evaluations]
    
    if st.button("开始对比"):
        comparison_result = backend_client.compare_evaluations(selected_ids)
        if comparison_result:
            show_comparison_results(comparison_result)
        else:
            st.error("对比失败")

def show_comparison_results(comparison_result):
    st.subheader("📈 对比结果")
    
    evaluations = comparison_result['evaluations']
    
    # Extract metrics for comparison
    metrics_data = []
    for eval_data in evaluations:
        results = eval_data['results_summary']
        row = {
            'name': eval_data['name'],
            'timestamp': eval_data['timestamp'][:16],
            'llm_provider': eval_data['llm_provider'],
            'overall_score': results.get('final_score', 0),
            'sample_count': results.get('sample_count', 0)
        }
        
        # Add individual metric scores
        for metric in results.get('metrics', []):
            row[metric['name']] = metric['score']
        
        metrics_data.append(row)
    
    df_comparison = pd.DataFrame(metrics_data)
    
    # Display comparison table
    st.subheader("📋 详细对比表")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualizations
    st.subheader("📊 可视化对比")
    
    # Overall score comparison
    fig_overall = px.bar(
        df_comparison,
        x='name',
        y='overall_score',
        title='总体得分对比',
        color='name'
    )
    st.plotly_chart(fig_overall, use_container_width=True)
    
    # Individual metrics comparison (radar chart)
    metric_columns = [col for col in df_comparison.columns 
                     if col not in ['name', 'timestamp', 'llm_provider', 'overall_score']]
    
    if metric_columns:
        st.subheader("🎯 各项指标对比（雷达图）")
        
        fig_radar = go.Figure()
        
        for _, row in df_comparison.iterrows():
            values = [row[metric] for metric in metric_columns]
            values.append(values[0])  # Close the radar chart
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_columns + [metric_columns[0]],
                fill='toself',
                name=row['name']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="各项指标对比"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Download comparison results
    st.subheader("📥 导出对比结果")
    
    csv_data = df_comparison.to_csv(index=False)
    st.download_button(
        label="下载CSV格式",
        data=csv_data,
        file_name=f"evaluation_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    json_data = json.dumps(comparison_result, indent=2, ensure_ascii=False)
    st.download_button(
        label="下载JSON格式",
        data=json_data,
        file_name=f"evaluation_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

def show_new_evaluation():
    st.header("🆕 新建评估")
    st.info("点击下方按钮跳转到评估页面")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("标准评估流程", type="primary"):
            st.switch_page("app.py")
    
    with col2:
        if st.button("两步式工作流", type="secondary"):
            st.switch_page("app_two_step.py")

if __name__ == "__main__":
    main() 
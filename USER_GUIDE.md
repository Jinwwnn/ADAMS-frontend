# ADAMS Web App User Guide

## Adaptive Domain-Aware Metric Selection

Welcome to ADAMS! This comprehensive guide will walk you through every feature of our RAG (Retrieval-Augmented Generation) evaluation platform step by step.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [System Overview](#system-overview)
3. [Data Annotation Pipeline](#data-annotation-pipeline)
4. [Evaluation &amp; Weight Adjustment](#evaluation--weight-adjustment)
5. [History &amp; Comparison](#history--comparison)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

---

## 🚀 Getting Started

### Prerequisites

- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Data**: CSV or JSON files with question-answer datasets
- **Internet Connection**: Required for AI-powered evaluations

### Accessing ADAMS

1. Open your web browser
2. Navigate to: `http://localhost:8501`
3. You should see the ADAMS interface with a dark/light theme based on your browser settings

### Interface Overview

The ADAMS interface consists of:

- **Header**: Title and backend connection status
- **Navigation Tabs**: Three main sections (Data Annotation, Evaluation, History)
- **Sidebar**: System status and quick actions
- **Main Area**: Context-specific tools and results

---

## 🔧 System Overview

### Three Main Sections

#### 1. **Data Annotation** 📊

Transform raw datasets by adding synthetic errors and key points for comprehensive evaluation.

#### 2. **Evaluation** 🧠

Run AI-powered evaluations with dynamic metric selection and real-time weight adjustment.

#### 3. **History** 📈

Review past evaluations and perform detailed comparisons between different runs.

### Status Indicators

- **🟢 Backend API Connected**: Full functionality available
- **🟡 Backend API Unavailable**: Limited functionality
- **🔴 Backend API Disconnected**: Check backend service

---

## 📊 Data Annotation Pipeline

### Purpose

The Data Annotation Pipeline enhances your dataset by:

- Adding key points extraction
- Introducing controlled synthetic errors
- Generating alternative answers for comprehensive testing

### Step-by-Step Process

#### Step 1: Upload Your Dataset

1. Click the **Data Annotation** tab
2. Look for the "📂 Dataset Upload" section
3. Click **"Browse files"** or drag and drop your file
4. **Recommended formats**: CSV
5. **Required columns**: `question`, `response`, `documents`

#### Step 2: Review Dataset Preview

1. After upload, check the **"📋 Data Preview"** section
2. Verify your data loaded correctly
3. Check the dataset statistics (samples, columns, file size)
4. Ensure all required fields are present

#### Step 3: API Status Check

1. In the **"🔗 API Status Check"** section:
   - **Backend API**: Should show 🟢 Connected
   - **OpenAI API**: Click "🔍 Test OpenAI API" to verify connectivity
2. If APIs are not connected, see [Troubleshooting](#troubleshooting)

#### Step 4: Run Annotation Pipeline

1. Click **"🚀 Run Data Annotation Pipeline to add mistakes"**
2. **Processing time**: 1-10 minutes depending on dataset size

#### Step 5: Review Annotated Results

1. Check the **"📊 Pipeline Results"** section for statistics
2. Review new columns added to your dataset
3. Use the **"👀 Dataset Preview"** slider to examine results
4. Download your enhanced dataset in CSV format

### Tips for Best Results

- **Data Quality**: Ensure clean, well-formatted input data
- **Size**: Start with smaller datasets (10-100 samples) for testing

---

## 🧠 Evaluation & Weight Adjustment

### Purpose

The Evaluation system uses AI agents to:

- Automatically select the most appropriate evaluation metrics
- Determine optimal weights for each metric
- Provide real-time score adjustments
- Generate detailed evaluation reports

### Step-by-Step Process

#### Step 1: Prepare Your Dataset

1. Click the **Evaluation** tab
2. Upload your evaluation dataset or use previously annotated data
3. **Required columns**: `question`, `response`, `documents`, `key_points`
4. Review the dataset preview to ensure data integrity

#### Step 2: Configure Evaluation Settings

1. **Select LLM Model**:

   - `gpt-4o-mini` (Recommended for speed and cost)

#### Step 3: Start Agent-based Evaluation

1. Click **"🚀 Start Agent-based Evaluation"**
2. **Progress monitoring**:
   - AI agents will discuss and refine metric selection
   - Real-time progress updates show current stage
   - **Estimated time**: 5-15 minutes for comprehensive evaluation
3. **What happens**:
   - Agents analyze your dataset characteristics
   - Optimal metric weights are determined
   - Final scores are calculated and aggregated

#### Step 4: Review Evaluation Results

##### Final Score Display

- **Large numerical score**: The overall quality rating

##### Individual Metric Scores

1. Expand each evaluator section to see:
   - **Average Score**: Mean performance across samples
   - **Standard Deviation**: Score consistency
   - **Min/Max Values**: Performance range
   - **Sample Scores**: Examples from your dataset

##### AI Discussion Summary

- **Rationale**: Why specific metrics were chosen
- **Insights**: Key findings about your dataset
- **Recommendations**: Suggestions for improvement

#### Step 5: Adjust Weights in Real-time

##### Using the Weight Adjustment Panel

1. Find the **"⚖️ Weight Adjustment"** section
2. **Drag sliders** to modify metric importance
3. **Real-time updates**: Scores recalculate immediately
4. **Visual feedback**: Progress bars show current weight distribution

### Best Practices

- **Start with AI recommendations**: The agents choose weights based on your data
- **Adjust gradually**: Small changes often have significant impact
- **Focus on priorities**: Increase weights for metrics most important to your use case
- **Normalize regularly**: Keep total weights at 100% for accurate comparisons

---

## 📈 History & Comparison

### Purpose

The History system allows you to:

- Track all previous evaluations
- Compare performance across different runs

### Viewing Evaluation History

#### Step 1: Access History

1. Click the **History** tab
2. View the **"📋 Evaluation History Records"** table with:
   - **ID**: Unique evaluation identifier
   - **Name**: Descriptive evaluation name
   - **Dataset Type**: Source of evaluated data
   - **LLM Judge**: Model used for evaluation
   - **Sample Count**: Number of samples evaluated
   - **Average Score**: Overall performance rating
   - **Evaluation Time**: When the evaluation was performed

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Backend Connection Problems

**Symptoms**: 🟡 Backend API Unavailable
**Solutions**:

1. Check if backend service is running on port 8000
2. Verify network connectivity
3. Restart backend service if needed

#### OpenAI API Issues

**Symptoms**: Evaluation fails, API key errors
**Solutions**:

1. Verify OPENAI_API_KEY environment variable is set
2. Check API key validity and remaining credits
3. Test connectivity: curl https://api.openai.com/v1/models
4. Try different model options if one is unavailable

---

Thank you for using ADAMS! This powerful evaluation platform is designed to help you assess and improve your RAG systems with confidence. Start with small datasets, experiment with features, and gradually scale up to comprehensive evaluations.

**Happy evaluating! 🚀**

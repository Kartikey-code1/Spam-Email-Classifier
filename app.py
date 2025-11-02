"""
Professional Spam Email Classifier Dashboard
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import time
import io

# Import our custom modules
from spam_classifier import SpamClassifier
from data_processor import TextPreprocessor
from sample_data import get_sample_data

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-spam {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
        animation: pulse 2s infinite;
    }
    .prediction-ham {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(81,207,102,0.3);
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    .stats-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None

def auto_train_model():
    """Auto-train model on first load"""
    if not st.session_state.model_trained:
        with st.spinner("🤖 Initializing AI Model... Please wait"):
            emails, labels = get_sample_data()
            classifier = SpamClassifier(model_type='naive_bayes')
            metrics = classifier.train(emails, labels, test_size=0.2)
            
            st.session_state.classifier = classifier
            st.session_state.model_trained = True
            st.session_state.training_metrics = metrics

def main():
    # Auto-train model
    auto_train_model()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Spam Email Classifier Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Email Security & Fraud Detection System</p>', unsafe_allow_html=True)
    
    # Quick stats display
    if st.session_state.training_metrics:
        col1, col2, col3, col4 = st.columns(4)
        metrics = st.session_state.training_metrics
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Accuracy</h3>
                <h2>{metrics['accuracy']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍 Precision</h3>
                <h2>{metrics['precision']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 F1 Score</h3>
                <h2>{metrics['f1_score']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Status</h3>
                <h2>Ready</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Email Check", "📁 CSV Bulk Analysis", "📊 Dashboard Analytics"])
    
    with tab1:
        show_single_email_check()
    
    with tab2:
        show_csv_upload()
    
    with tab3:
        show_dashboard_analytics()

def show_single_email_check():
    """Single email classification interface"""
    st.markdown("### 📧 Type or Paste Email Content")
    
    # Text input area
    email_text = st.text_area(
        "Enter email content to analyze:",
        height=200,
        placeholder="Type or paste the email content here...\n\nExample:\n'URGENT! You have won $1000000! Click here to claim your prize now!'\n\nOr:\n'Hi John, can we schedule a meeting for tomorrow at 2 PM?'",
        key="email_input"
    )
    
    # Quick sample buttons
    st.markdown("**Quick Test Samples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 Spam Example", use_container_width=True):
            st.session_state.email_input = "URGENT! You have won $1000000! Click here to claim your prize now! Limited time offer!"
            st.rerun()
    
    with col2:
        if st.button("✅ Ham Example", use_container_width=True):
            st.session_state.email_input = "Hi John, can we schedule a meeting for tomorrow at 2 PM? Please let me know your availability."
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.email_input = ""
            st.rerun()
    
    # Analysis button
    if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
        if email_text.strip():
            analyze_single_email(email_text)
        else:
            st.error("⚠️ Please enter an email to analyze.")

def analyze_single_email(email_text):
    """Analyze a single email"""
    with st.spinner("🤖 AI is analyzing the email..."):
        time.sleep(0.5)  # Visual feedback
        result = st.session_state.classifier.predict(email_text)
        
        # Display result with animation
        st.markdown("### 📋 Analysis Result")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if result['prediction'] == 'spam':
                st.markdown(f"""
                <div class="prediction-spam">
                    🚨 SPAM DETECTED<br>
                    <h2>Confidence: {result['spam_probability']:.1%}</h2>
                    <p>This email appears to be fraudulent or spam</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-ham">
                    ✅ LEGITIMATE EMAIL<br>
                    <h2>Confidence: {result['ham_probability']:.1%}</h2>
                    <p>This email appears to be safe and legitimate</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Probability gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['spam_probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Spam Risk %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if result['spam_probability'] > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_csv_upload():
    """CSV upload and bulk analysis interface"""
    st.markdown("### 📁 Upload CSV File for Bulk Email Analysis")
    
    st.markdown("""
    <div class="upload-section">
        <h3>📋 CSV Format Requirements:</h3>
        <ul>
            <li>CSV file should have a column named 'email' or 'text' or 'message'</li>
            <li>Each row should contain one email content</li>
            <li>Maximum file size: 200MB</li>
            <li>Supported formats: .csv</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing email content for bulk analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows.")
            
            # Show preview
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Identify email column
            email_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['email', 'text', 'message', 'content'])]
            
            if email_columns:
                email_column = st.selectbox("Select email content column:", email_columns)
                
                # Analysis button
                if st.button("🚀 Analyze All Emails", type="primary", use_container_width=True):
                    analyze_csv_emails(df, email_column)
            else:
                st.error("⚠️ No email content column found. Please ensure your CSV has a column named 'email', 'text', 'message', or 'content'.")
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

def analyze_csv_emails(df, email_column):
    """Analyze emails from CSV"""
    st.markdown("### 🤖 Bulk Analysis Results")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, email_text in enumerate(df[email_column]):
        if pd.notna(email_text) and str(email_text).strip():
            result = st.session_state.classifier.predict(str(email_text))
            results.append({
                'Email': str(email_text)[:100] + "..." if len(str(email_text)) > 100 else str(email_text),
                'Prediction': result['prediction'],
                'Spam_Probability': result['spam_probability'],
                'Status': '🚨 SPAM' if result['prediction'] == 'spam' else '✅ HAM'
            })
        else:
            results.append({
                'Email': 'Empty or invalid',
                'Prediction': 'unknown',
                'Spam_Probability': 0,
                'Status': '❓ UNKNOWN'
            })
        
        # Update progress
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing email {i+1} of {len(df)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    spam_count = len(results_df[results_df['Prediction'] == 'spam'])
    ham_count = len(results_df[results_df['Prediction'] == 'ham'])
    total_count = len(results_df)
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stats-container">
            <h3>📧 Total Emails</h3>
            <h2>{total_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-container">
            <h3>🚨 Spam Detected</h3>
            <h2>{spam_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-container">
            <h3>✅ Legitimate</h3>
            <h2>{ham_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        spam_rate = (spam_count / total_count * 100) if total_count > 0 else 0
        st.markdown(f"""
        <div class="stats-container">
            <h3>📊 Spam Rate</h3>
            <h2>{spam_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Results table
    st.markdown("#### 📋 Detailed Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Download results
    csv_buffer = io.StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="spam_analysis_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Visualization
    if spam_count > 0 or ham_count > 0:
        fig = px.pie(
            values=[spam_count, ham_count],
            names=['Spam', 'Ham'],
            title="Email Classification Distribution",
            color_discrete_map={'Spam': '#ff6b6b', 'Ham': '#51cf66'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_dashboard_analytics():
    """Dashboard analytics and insights"""
    st.markdown("### 📊 Model Performance & Analytics")
    
    if st.session_state.training_metrics:
        metrics = st.session_state.training_metrics
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Performance Metrics")
            
            # Radar chart
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance',
                line_color='#1f77b4'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                title="Model Performance Radar Chart",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📈 Confusion Matrix")
            
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
        
        # Feature importance
        st.markdown("#### 🔍 Top Spam Indicators")
        
        feature_importance = st.session_state.classifier.get_feature_importance(top_n=15)
        
        if feature_importance:
            features_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
            features_df = features_df.head(10)
            
            fig = px.bar(
                features_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Most Important Spam Detection Features',
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Word clouds
        st.markdown("#### ☁️ Word Analysis")
        
        emails, labels = get_sample_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🚨 Common Spam Words")
            spam_emails = [emails[i] for i, label in enumerate(labels) if label == 'spam']
            spam_text = ' '.join(spam_emails)
            
            if spam_text:
                wordcloud_spam = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Reds',
                    max_words=50
                ).generate(spam_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_spam, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        
        with col2:
            st.markdown("##### ✅ Common Legitimate Words")
            ham_emails = [emails[i] for i, label in enumerate(labels) if label == 'ham']
            ham_text = ' '.join(ham_emails)
            
            if ham_text:
                wordcloud_ham = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Greens',
                    max_words=50
                ).generate(ham_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_ham, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

if __name__ == "__main__":
    main()
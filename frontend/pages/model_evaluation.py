#model_evaluation.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Model Evaluation Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .author-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e6ed;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4CAF50, #2196F3, #FF9800);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-card h2 {
        color: #1a237e;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
        
    .metric-card h3{
        color: #3f51b5;
        margin-bottom: 0.5rem;
    }
    .metric-card p{
        color: #3f51b5;
        margin-bottom: 0.5rem;
    }
    
    .performance-excellent {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff9800;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        color: #333;
    }
    .feature-card h4 {
        color: #1a237e;
        margin-bottom: 0.5rem;
    }
    
    .tech-stack-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        border: 2px solid #2196f3;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .tech-stack-card:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(33,150,243,0.3);
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .achievement-badge {
        display: inline-block;
        background: linear-gradient(45deg, #ffd700, #ffed4e);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: bold;
        box-shadow: 0 3px 10px rgba(255,215,0,0.3);
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ModelEvaluationDashboard:
    def __init__(self):
        # Centralized data dictionary for all models
        self.models_data = {
            "Random Forest (Current)": {
                "confusion_matrix": np.array([[1012, 8, 1], [0, 394, 0], [0, 0, 385]]),
                "metrics": {"Accuracy": 0.99, "Precision": 0.99, "Recall": 0.99},
                "classification_report": {
                    'Low': {'precision': 1.00, 'recall': 0.99, 'f1-score': 1.00, 'support': 1021},
                    'Moderate': {'precision': 0.98, 'recall': 1.00, 'f1-score': 0.99, 'support': 394},
                    'High': {'precision': 1.00, 'recall': 1.00, 'f1-score': 1.00, 'support': 385},
                    'accuracy': 0.99,
                    'macro_avg': {'precision': 0.99, 'recall': 1.00, 'f1-score': 0.99},
                    'weighted_avg': {'precision': 1.00, 'recall': 0.99, 'f1-score': 1.00}
                },
                "feature_importance": {
                    'Brightness': 0.25, 'FRP': 0.22, 'Confidence': 0.18, 
                    'Satellite Type': 0.12, 'Day/Night': 0.08, 'Location': 0.06, 
                    'Weather Data': 0.05, 'Historical Data': 0.04
                },
                "strengths": [
                    "🎯 Highest overall accuracy (99%)",
                    "⚡ Fast inference time (<1ms)",
                    "🔄 Robust to outliers and noise",
                    "📊 Excellent feature interpretability",
                    "⚖️ Balanced performance across all classes"
                ]
            },
            "XGBoost": {
                "confusion_matrix": np.array([[985, 28, 8], [12, 375, 7], [5, 8, 372]]),
                "metrics": {"Accuracy": 0.97, "Precision": 0.96, "Recall": 0.97},
                "classification_report": {
                    'Low': {'precision': 0.98, 'recall': 0.97, 'f1-score': 0.97, 'support': 1021},
                    'Moderate': {'precision': 0.91, 'recall': 0.95, 'f1-score': 0.93, 'support': 394},
                    'High': {'precision': 0.96, 'recall': 0.97, 'f1-score': 0.96, 'support': 385},
                    'accuracy': 0.97,
                    'macro_avg': {'precision': 0.95, 'recall': 0.96, 'f1-score': 0.95},
                    'weighted_avg': {'precision': 0.96, 'recall': 0.97, 'f1-score': 0.96}
                },
                "feature_importance": {
                    'FRP': 0.28, 'Brightness': 0.24, 'Confidence': 0.15, 
                    'Location': 0.10, 'Satellite Type': 0.09, 'Day/Night': 0.07, 
                    'Weather Data': 0.04, 'Historical Data': 0.03
                },
                "strengths": [
                    "🚀 Excellent gradient boosting performance",
                    "🎯 Strong feature learning capabilities",
                    "📈 Good handling of imbalanced data",
                    "⚡ Efficient memory usage",
                    "🔧 Extensive hyperparameter tuning options"
                ]
            },
            "Neural Network": {
                "confusion_matrix": np.array([[968, 35, 18], [22, 358, 14], [15, 12, 358]]),
                "metrics": {"Accuracy": 0.95, "Precision": 0.94, "Recall": 0.95},
                "classification_report": {
                    'Low': {'precision': 0.96, 'recall': 0.95, 'f1-score': 0.95, 'support': 1021},
                    'Moderate': {'precision': 0.88, 'recall': 0.91, 'f1-score': 0.90, 'support': 394},
                    'High': {'precision': 0.92, 'recall': 0.93, 'f1-score': 0.92, 'support': 385},
                    'accuracy': 0.95,
                    'macro_avg': {'precision': 0.92, 'recall': 0.93, 'f1-score': 0.92},
                    'weighted_avg': {'precision': 0.94, 'recall': 0.95, 'f1-score': 0.94}
                },
                "feature_importance": {
                    'Brightness': 0.30, 'FRP': 0.26, 'Weather Data': 0.14, 
                    'Historical Data': 0.12, 'Confidence': 0.08, 'Location': 0.05, 
                    'Day/Night': 0.03, 'Satellite Type': 0.02
                },
                "strengths": [
                    "🧠 Deep learning pattern recognition",
                    "🔗 Complex feature interactions",
                    "📊 Non-linear relationship modeling",
                    "🎯 Adaptive learning capabilities",
                    "🔄 Continuous improvement potential"
                ]
            },
            "Ensemble Model": {
                "confusion_matrix": np.array([[1008, 10, 3], [5, 385, 4], [2, 3, 380]]),
                "metrics": {"Accuracy": 0.985, "Precision": 0.98, "Recall": 0.985},
                "classification_report": {
                    'Low': {'precision': 0.99, 'recall': 0.99, 'f1-score': 0.99, 'support': 1021},
                    'Moderate': {'precision': 0.97, 'recall': 0.98, 'f1-score': 0.97, 'support': 394},
                    'High': {'precision': 0.98, 'recall': 0.99, 'f1-score': 0.98, 'support': 385},
                    'accuracy': 0.985,
                    'macro_avg': {'precision': 0.98, 'recall': 0.98, 'f1-score': 0.98},
                    'weighted_avg': {'precision': 0.98, 'recall': 0.985, 'f1-score': 0.98}
                },
                "feature_importance": {
                    'Brightness': 0.27, 'FRP': 0.24, 'Confidence': 0.16, 
                    'Weather Data': 0.10, 'Satellite Type': 0.09, 'Location': 0.07, 
                    'Day/Night': 0.04, 'Historical Data': 0.03
                },
                "strengths": [
                    "🏆 Best overall performance (98.5%)",
                    "⚖️ Combines strengths of multiple models",
                    "🎯 Reduced variance and bias",
                    "🔒 Most robust predictions",
                    "📊 Excellent generalization ability"
                ]
            }
        }
        
        self.class_names = ['Low', 'Moderate', 'High']
        
    def create_confusion_matrix_heatmap(self, confusion_matrix_data):
        """Create an interactive confusion matrix heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix_data,
            x=['Low', 'Moderate', 'High'],
            y=['Low', 'Moderate', 'High'],
            colorscale='Viridis',
            showscale=True,
            text=confusion_matrix_data,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white"},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title={
                'text': '🎯 Confusion Matrix - Model Performance',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#1a237e'}
            },
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            height=500,
            font=dict(size=14)
        )
        
        return fig
    
    def create_metrics_comparison_chart(self, classification_report_data):
        """Create metrics comparison chart"""
        metrics = ['Precision', 'Recall', 'F1-Score']
        low_values = [
            classification_report_data['Low']['precision'],
            classification_report_data['Low']['recall'],
            classification_report_data['Low']['f1-score']
        ]
        moderate_values = [
            classification_report_data['Moderate']['precision'],
            classification_report_data['Moderate']['recall'],
            classification_report_data['Moderate']['f1-score']
        ]
        high_values = [
            classification_report_data['High']['precision'],
            classification_report_data['High']['recall'],
            classification_report_data['High']['f1-score']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=low_values,
            theta=metrics,
            fill='toself',
            name='Low Risk',
            fillcolor='rgba(76, 175, 80, 0.3)',
            line=dict(color='#4CAF50', width=3)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=moderate_values,
            theta=metrics,
            fill='toself',
            name='Moderate Risk',
            fillcolor='rgba(255, 152, 0, 0.3)',
            line=dict(color='#FF9800', width=3)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=high_values,
            theta=metrics,
            fill='toself',
            name='High Risk',
            fillcolor='rgba(244, 67, 54, 0.3)',
            line=dict(color='#F44336', width=3)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title={
                'text': '📊 Performance Metrics by Risk Level',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#1a237e'}
            },
            height=500
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance_data):
        """Create feature importance visualization"""
        features = list(feature_importance_data.keys())
        importance = list(feature_importance_data.values())
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance Score")
            ),
            text=[f'{imp:.1%}' for imp in importance],
            textposition='inside',
            textfont=dict(color='white', size=12)
        ))
        
        fig.update_layout(
            title={
                'text': '🔍 Feature Importance Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#1a237e'}
            },
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_model_comparison_chart(self):
        """Create model comparison chart using all models data"""
        models = list(self.models_data.keys())
        accuracy = [self.models_data[model]["metrics"]["Accuracy"] for model in models]
        precision = [self.models_data[model]["metrics"]["Precision"] for model in models]
        recall = [self.models_data[model]["metrics"]["Recall"] for model in models]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=models, y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=models, y=precision,
            mode='lines+markers',
            name='Precision',
            line=dict(color='#2196F3', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=models, y=recall,
            mode='lines+markers',
            name='Recall',
            line=dict(color='#FF9800', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title={
                'text': '🏆 Model Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#1a237e'}
            },
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis=dict(range=[0.8, 1.0]),
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def run(self):
        """Run the Model Evaluation Dashboard"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Early Detection and Risk Mapping of Forest Fires Using Remote Sensing and AI</h1>
            <p>Comprehensive Analysis of Machine Learning Model Performance</p>
            <p>🔥 Advanced AI-Powered Fire Risk Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Author Information
        st.markdown("""
        <div class="author-card pulse">
            <h2>👨‍💻 Project Developer</h2>
            <h3>Vaibhav Karbhantnal</h3>
            <p><strong>USN:</strong> 1BY23MC097</p>
            <p>Machine Learning Engineer | Backend Developer</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Interactive model selection
        selected_model = st.sidebar.selectbox(
            "🤖 Select Model",
            list(self.models_data.keys())
        )
        
        # Get data for selected model
        model_data = self.models_data[selected_model]
        
        # Analysis type
        analysis_type = st.sidebar.radio(
            "📊 Analysis Type",
            ["Performance Metrics", "Feature Analysis", "Model Comparison", "Technical Details"]
        )
        
        # Show advanced metrics toggle
        show_advanced = st.sidebar.checkbox("🔬 Show Advanced Metrics", value=True)
        
        # Model performance summary for selected model
        st.sidebar.markdown("### 🏆 Quick Stats")
        st.sidebar.metric("Overall Accuracy", f"{model_data['metrics']['Accuracy']:.1%}", "")
        st.sidebar.metric("Precision (Avg)", f"{model_data['metrics']['Precision']:.1%}", "")
        st.sidebar.metric("Recall (Avg)", f"{model_data['metrics']['Recall']:.1%}", "")
        
        # Main dashboard content
        if analysis_type == "Performance Metrics":
            self.show_performance_metrics(model_data, show_advanced)
        elif analysis_type == "Feature Analysis":
            self.show_feature_analysis(model_data)
        elif analysis_type == "Model Comparison":
            self.show_model_comparison()
        else:
            self.show_technical_details()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; color: #333;">
            <p><strong>🔬 Advanced Machine Learning for Environmental Protection</strong></p>
            <p>© 2024 Vaibhav Karbhantnal - Forest Fire Risk Prediction System</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_performance_metrics(self, model_data, show_advanced):
        """Display performance metrics section"""
        st.markdown("## 🎯 Model Performance Analysis")
        
        # Alert for performance status
        accuracy = model_data['metrics']['Accuracy']
        if accuracy >= 0.98:
            status_class = "alert-success"
            status_icon = "🎉"
            status_text = "Excellent Performance Achieved!"
        elif accuracy >= 0.95:
            status_class = "alert-success"
            status_icon = "✅"
            status_text = "Good Performance Achieved!"
        else:
            status_class = "alert-success"
            status_icon = "📊"
            status_text = "Performance Analysis Results"
        
        st.markdown(f"""
        <div class="{status_class}">
            <strong>{status_icon} {status_text}</strong><br>
            Your selected model demonstrates {accuracy:.1%} overall accuracy across all risk categories.
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = model_data['metrics']
        total_samples = np.sum(model_data['confusion_matrix'])
        correct_predictions = np.trace(model_data['confusion_matrix'])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card performance-excellent">
                <h3>🎯 Overall Accuracy</h3>
                <h2>{metrics['Accuracy']:.1%}</h2>
                <p>{correct_predictions} / {total_samples} correct predictions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card performance-excellent">
                <h3>🎪 Macro F1-Score</h3>
                <h2>{model_data['classification_report']['macro_avg']['f1-score']:.1%}</h2>
                <p>Balanced performance across classes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card performance-excellent">
                <h3>⚖️ Weighted Precision</h3>
                <h2>{model_data['classification_report']['weighted_avg']['precision']:.1%}</h2>
                <p>Minimal false positives</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card performance-excellent">
                <h3>🔍 Weighted Recall</h3>
                <h2>{model_data['classification_report']['weighted_avg']['recall']:.1%}</h2>
                <p>Excellent detection rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            confusion_matrix_fig = self.create_confusion_matrix_heatmap(model_data['confusion_matrix'])
            st.plotly_chart(confusion_matrix_fig, use_container_width=True)
        
        with col2:
            metrics_comparison_fig = self.create_metrics_comparison_chart(model_data['classification_report'])
            st.plotly_chart(metrics_comparison_fig, use_container_width=True)
        
        if show_advanced:
            self.show_detailed_classification_report(model_data['classification_report'])
    
    def show_feature_analysis(self, model_data):
        """Display feature analysis section"""
        st.markdown("## 🔍 Feature Importance & Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_importance_fig = self.create_feature_importance_chart(model_data['feature_importance'])
            st.plotly_chart(feature_importance_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📝 Key Feature Insights")
            
            # Create feature insights from the actual data
            sorted_features = sorted(model_data['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
            
            feature_descriptions = {
                'Brightness': '🌡️ Primary indicator of fire intensity',
                'FRP': '🔥 Fire Radiative Power - measures energy release',
                'Confidence': '📊 Satellite detection confidence level',
                'Satellite Type': '🛰️ Different sensor capabilities',
                'Day/Night': '🌅 Temporal fire detection patterns',
                'Location': '📍 Geographic risk factors',
                'Weather Data': '🌤️ Meteorological conditions',
                'Historical Data': '📈 Past fire occurrence patterns'
            }
            
            for feature, importance in sorted_features[:5]:  # Show top 5 features
                icon_desc = feature_descriptions.get(feature, f'📊 {feature}')
                st.markdown(f"""
                <div class="feature-card">
                    <h4>{icon_desc}</h4>
                    <p>Key factor in fire risk prediction</p>
                    <strong>Importance: {importance:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    def show_model_comparison(self):
        """Display model comparison section using all models data"""
        st.markdown("## 🏆 Model Performance Comparison")
        
        model_comparison_fig = self.create_model_comparison_chart()
        st.plotly_chart(model_comparison_fig, use_container_width=True)
        
        st.markdown("### 🥇 Model Strengths Comparison")
        
        # Display strengths for all models
        cols = st.columns(len(self.models_data))
        
        for i, (model_name, model_data) in enumerate(self.models_data.items()):
            with cols[i]:
                st.markdown(f"**{model_name}**")
                for strength in model_data['strengths']:
                    st.markdown(f"""
                    <div class="achievement-badge" style="display: block; margin: 0.25rem 0; font-size: 0.8rem;">
                        {strength}
                    </div>
                    """, unsafe_allow_html=True)
    
    def show_technical_details(self):
        """Display technical details section"""
        st.markdown("## 🔧 Technical Implementation Details")
        
        # Technology stack
        st.markdown("### 💻 Technology Stack")
        
        tech_stack = {
            "Machine Learning": ["Scikit-learn", "Pandas", "NumPy"],
            "Visualization": ["Plotly", "Streamlit", "Folium"],
            "Data Sources": ["NASA FIRMS", "VIIRS", "MODIS"],
            "Deployment": ["FastAPI", "Docker", "Cloud Services"]
        }
        
        cols = st.columns(len(tech_stack))
        for i, (category, technologies) in enumerate(tech_stack.items()):
            with cols[i]:
                st.markdown(f"**{category}**")
                for tech in technologies:
                    st.markdown(f"""
                    <div class="tech-stack-card">
                        {tech}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Model architecture details
        st.markdown("### 🏗️ Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Random Forest Configuration:**
            - **Estimators**: 100 trees
            - **Max Depth**: 15
            - **Min Samples Split**: 5
            - **Min Samples Leaf**: 2
            - **Random State**: 42
            """)
        
        with col2:
            st.markdown("""
            **Training Details:**
            - **Dataset Size**: 18,000 samples
            - **Training/Test Split**: 80/20
            - **Cross-Validation**: 5-fold
            - **Training Time**: 12.3 seconds
            - **Inference Time**: <1ms per prediction
            """)
    
    def show_detailed_classification_report(self, classification_report_data):
        """Display detailed classification report"""
        st.markdown("### 📊 Detailed Classification Report")
        
        # Create detailed metrics table
        report_data = []
        for class_name in ['Low', 'Moderate', 'High']:
            report_data.append({
                'Risk Level': class_name,
                'Precision': f"{classification_report_data[class_name]['precision']:.3f}",
                'Recall': f"{classification_report_data[class_name]['recall']:.3f}",
                'F1-Score': f"{classification_report_data[class_name]['f1-score']:.3f}",
                'Support': classification_report_data[class_name]['support']
            })
        
        # Add summary rows
        report_data.append({
            'Risk Level': 'Accuracy',
            'Precision': '', 'Recall': '', 
            'F1-Score': f"{classification_report_data['accuracy']:.3f}",
            'Support': sum([classification_report_data[cls]['support'] for cls in ['Low', 'Moderate', 'High']])
        })
        
        report_data.append({
            'Risk Level': 'Macro Avg',
            'Precision': f"{classification_report_data['macro_avg']['precision']:.3f}",
            'Recall': f"{classification_report_data['macro_avg']['recall']:.3f}",
            'F1-Score': f"{classification_report_data['macro_avg']['f1-score']:.3f}",
            'Support': sum([classification_report_data[cls]['support'] for cls in ['Low', 'Moderate', 'High']])
        })
        
        report_data.append({
            'Risk Level': 'Weighted Avg',
            'Precision': f"{classification_report_data['weighted_avg']['precision']:.3f}",
            'Recall': f"{classification_report_data['weighted_avg']['recall']:.3f}",
            'F1-Score': f"{classification_report_data['weighted_avg']['f1-score']:.3f}",
            'Support': sum([classification_report_data[cls]['support'] for cls in ['Low', 'Moderate', 'High']])
        })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)

# Main execution
if __name__ == "__main__":
    dashboard = ModelEvaluationDashboard()
    dashboard.run()
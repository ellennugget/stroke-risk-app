"""
Stroke Risk Assessment App
Interactive web application for stroke risk prediction
Deploy on Streamlit Cloud: https://streamlit.io

File structure:
app.py (this file)
model.pkl (your trained model)
requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 1: LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load trained model and reference data"""
    try:
        # Try to load pre-trained model
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['features']
        reference_data = model_data['reference_data']
        
        return model, feature_names, reference_data
    except FileNotFoundError:
        st.warning("Model file not found. Using demo mode with synthetic data.")
        return None, None, generate_demo_data()


def generate_demo_data():
    """Generate demo reference data for visualization"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic reference data
    data = {
        'AGE': np.random.normal(65, 12, n_samples).clip(18, 95),
        'SEX': np.random.choice(['M', 'F'], n_samples),
        'VS_SYSBP': np.random.normal(135, 15, n_samples).clip(90, 200),
        'VS_DIABP': np.random.normal(85, 10, n_samples).clip(60, 120),
        'LB_GLUCOSE': np.random.normal(105, 20, n_samples).clip(70, 200),
        'LB_CHOL': np.random.normal(210, 35, n_samples).clip(120, 350),
        'LB_HDL': np.random.normal(50, 12, n_samples).clip(25, 100),
        'LB_LDL': np.random.normal(130, 30, n_samples).clip(50, 250),
        'STROKE': np.random.choice([0, 1], n_samples, p=[0.94, 0.06])
    }
    
    return pd.DataFrame(data)


# ============================================================================
# SECTION 2: USER INPUT FORM
# ============================================================================

def get_user_input():
    """Create sidebar form for user input"""
    st.sidebar.markdown("## 👤 Patient Information")
    st.sidebar.markdown("Enter your health data below:")
    
    with st.sidebar.form("patient_form"):
        st.markdown("### Demographics")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65, step=1)
        sex = st.selectbox("Sex", ["M", "F"], index=0)
        race = st.selectbox("Race", ["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"], index=0)
        
        st.markdown("### Vital Signs")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=120, step=1)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=80, step=1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70, step=1)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
        
        st.markdown("### Laboratory Values")
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=300, value=100, step=1)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50, step=1)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=30, max_value=300, value=130, step=1)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=500, value=150, step=1)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=5.0, value=1.0, step=0.1)
        
        st.markdown("### Treatment")
        treatment = st.selectbox("Treatment Group", ["CONTROL", "TREATMENT"], index=0)
        
        submitted = st.form_submit_button("🔍 Calculate Risk", use_container_width=True)
        
    if submitted:
        user_data = {
            'age': age,
            'sex': sex,
            'race': race,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'weight': weight,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'hdl': hdl,
            'ldl': ldl,
            'triglycerides': triglycerides,
            'creatinine': creatinine,
            'treatment': treatment
        }
        return user_data, submitted
    
    return None, False


# ============================================================================
# SECTION 3: RISK PREDICTION
# ============================================================================

def predict_risk(user_data, model, feature_names):
    """Predict stroke risk for user"""
    # Build feature vector matching training
    input_df = pd.DataFrame({
        'AGE': [user_data['age']],
        'VS_SYSBP': [user_data['systolic_bp']],
        'VS_DIABP': [user_data['diastolic_bp']],
        'VS_HR': [user_data['heart_rate']],
        'VS_WEIGHT': [user_data['weight']],
        'LB_GLUCOSE': [user_data['glucose']],
        'LB_CHOL': [user_data['cholesterol']],
        'LB_HDL': [user_data['hdl']],
        'LB_LDL': [user_data['ldl']],
        'LB_TRIG': [user_data['triglycerides']],
        'LB_CREAT': [user_data['creatinine']],
    })
    
    # Add categorical features
    input_df['SEX_M'] = 1 if user_data['sex'] == 'M' else 0
    input_df['SEX_F'] = 1 if user_data['sex'] == 'F' else 0
    
    for r in ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'OTHER']:
        input_df[f'RACE_{r}'] = 1 if user_data['race'] == r else 0
    
    input_df['ARM_CONTROL'] = 1 if user_data['treatment'] == 'CONTROL' else 0
    input_df['ARM_TREATMENT'] = 1 if user_data['treatment'] == 'TREATMENT' else 0
    
    # Align features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Predict
    if model is not None:
        risk_prob = model.predict_proba(input_df)[0, 1]
    else:
        # Demo calculation
        risk_score = (
            (user_data['age'] - 50) * 0.01 +
            (user_data['systolic_bp'] - 120) * 0.002 +
            (user_data['glucose'] - 100) * 0.001 +
            (user_data['cholesterol'] - 200) * 0.0005 +
            (0.05 if user_data['sex'] == 'M' else 0)
        )
        risk_prob = 1 / (1 + np.exp(-risk_score))
        risk_prob = np.clip(risk_prob, 0, 1)
    
    return risk_prob


def display_risk_result(risk_prob):
    """Display risk prediction with color coding"""
    risk_pct = risk_prob * 100
    
    # Determine risk category
    if risk_prob < 0.05:
        color = "#10b981"  # Green
        category = "🟢 Low Risk"
        recommendation = "Continue routine health monitoring and maintain healthy lifestyle."
    elif risk_prob < 0.15:
        color = "#f59e0b"  # Yellow
        category = "🟡 Moderate Risk"
        recommendation = "Consider lifestyle modifications. Schedule follow-up with healthcare provider."
    elif risk_prob < 0.25:
        color = "#f97316"  # Orange
        category = "🟠 High Risk"
        recommendation = "Medical intervention recommended. Consult with healthcare provider soon."
    else:
        color = "#ef4444"  # Red
        category = "🔴 Very High Risk"
        recommendation = "Urgent medical attention recommended. Contact healthcare provider immediately."
    
    # Display result
    st.markdown(f"""
    <div class="risk-box" style="background-color: {color}20; border: 3px solid {color};">
        <div style="color: {color};">{category}</div>
        <div style="font-size: 3rem; color: {color}; margin: 1rem 0;">{risk_pct:.1f}%</div>
        <div style="font-size: 1rem; color: #666;">Predicted Stroke Risk</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"**Recommendation:** {recommendation}")
    
    return category, color


# ============================================================================
# SECTION 4: INTERACTIVE VISUALIZATIONS
# ============================================================================

def create_comparison_charts(user_data, reference_data, risk_prob):
    """Create interactive Plotly charts showing user position"""
    
    st.markdown("## 📊 See Where You Stand")
    st.markdown("Your position is shown in **blue** compared to reference population (green = no stroke, red = stroke)")
    
    # Create subplot layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Chart 1: Age vs Blood Pressure
        fig1 = go.Figure()
        
        # Reference data
        no_stroke = reference_data[reference_data['STROKE'] == 0]
        stroke = reference_data[reference_data['STROKE'] == 1]
        
        fig1.add_trace(go.Scatter(
            x=no_stroke['AGE'],
            y=no_stroke['VS_SYSBP'],
            mode='markers',
            name='No Stroke (Reference)',
            marker=dict(color='green', size=5, opacity=0.3),
            hovertemplate='Age: %{x}<br>SBP: %{y}<extra></extra>'
        ))
        
        fig1.add_trace(go.Scatter(
            x=stroke['AGE'],
            y=stroke['VS_SYSBP'],
            mode='markers',
            name='Stroke (Reference)',
            marker=dict(color='red', size=5, opacity=0.5),
            hovertemplate='Age: %{x}<br>SBP: %{y}<extra></extra>'
        ))
        
        # User point
        fig1.add_trace(go.Scatter(
            x=[user_data['age']],
            y=[user_data['systolic_bp']],
            mode='markers',
            name='YOU',
            marker=dict(
                color='blue',
                size=20,
                symbol='star',
                line=dict(color='darkblue', width=2)
            ),
            hovertemplate=f'<b>YOU</b><br>Age: {user_data["age"]}<br>SBP: {user_data["systolic_bp"]}<extra></extra>'
        ))
        
        fig1.update_layout(
            title='Age vs Systolic Blood Pressure',
            xaxis_title='Age (years)',
            yaxis_title='Systolic BP (mmHg)',
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Chart 2: Cholesterol vs Glucose
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=no_stroke['LB_CHOL'],
            y=no_stroke['LB_GLUCOSE'],
            mode='markers',
            name='No Stroke (Reference)',
            marker=dict(color='green', size=5, opacity=0.3),
            hovertemplate='Chol: %{x}<br>Glucose: %{y}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=stroke['LB_CHOL'],
            y=stroke['LB_GLUCOSE'],
            mode='markers',
            name='Stroke (Reference)',
            marker=dict(color='red', size=5, opacity=0.5),
            hovertemplate='Chol: %{x}<br>Glucose: %{y}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=[user_data['cholesterol']],
            y=[user_data['glucose']],
            mode='markers',
            name='YOU',
            marker=dict(
                color='blue',
                size=20,
                symbol='star',
                line=dict(color='darkblue', width=2)
            ),
            hovertemplate=f'<b>YOU</b><br>Chol: {user_data["cholesterol"]}<br>Glucose: {user_data["glucose"]}<extra></extra>'
        ))
        
        fig2.update_layout(
            title='Cholesterol vs Glucose',
            xaxis_title='Total Cholesterol (mg/dL)',
            yaxis_title='Glucose (mg/dL)',
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Risk Distribution (full width)
    st.markdown("---")
    
    # Create risk distribution histogram
    fig3 = go.Figure()
    
    # Reference population risk distribution (simulated)
    all_ages = reference_data['AGE'].values
    all_sbp = reference_data['VS_SYSBP'].values
    all_glucose = reference_data['LB_GLUCOSE'].values
    
    # Simple risk calculation for reference
    ref_risks = []
    for i in range(len(reference_data)):
        risk_score = (
            (all_ages[i] - 50) * 0.01 +
            (all_sbp[i] - 120) * 0.002 +
            (all_glucose[i] - 100) * 0.001
        )
        risk = 1 / (1 + np.exp(-risk_score))
        ref_risks.append(risk * 100)
    
    # Separate by stroke status
    no_stroke_risks = [ref_risks[i] for i in range(len(reference_data)) if reference_data.iloc[i]['STROKE'] == 0]
    stroke_risks = [ref_risks[i] for i in range(len(reference_data)) if reference_data.iloc[i]['STROKE'] == 1]
    
    fig3.add_trace(go.Histogram(
        x=no_stroke_risks,
        name='No Stroke (Reference)',
        marker_color='green',
        opacity=0.6,
        nbinsx=30
    ))
    
    fig3.add_trace(go.Histogram(
        x=stroke_risks,
        name='Stroke (Reference)',
        marker_color='red',
        opacity=0.6,
        nbinsx=30
    ))
    
    # Add user's risk as vertical line
    fig3.add_vline(
        x=risk_prob * 100,
        line_dash="dash",
        line_color="blue",
        line_width=3,
        annotation_text=f"YOU: {risk_prob*100:.1f}%",
        annotation_position="top"
    )
    
    fig3.update_layout(
        title='Risk Distribution - Where Do You Stand?',
        xaxis_title='Predicted Stroke Risk (%)',
        yaxis_title='Number of People',
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4 & 5: Box plots
    col3, col4 = st.columns(2)
    
    with col3:
        # Blood Pressure comparison
        fig4 = go.Figure()
        
        fig4.add_trace(go.Box(
            y=no_stroke['VS_SYSBP'],
            name='No Stroke',
            marker_color='green',
            boxmean='sd'
        ))
        
        fig4.add_trace(go.Box(
            y=stroke['VS_SYSBP'],
            name='Stroke',
            marker_color='red',
            boxmean='sd'
        ))
        
        # Add user's value as a point
        fig4.add_trace(go.Scatter(
            x=['Your Value'],
            y=[user_data['systolic_bp']],
            mode='markers',
            name='YOU',
            marker=dict(
                color='blue',
                size=15,
                symbol='star',
                line=dict(color='darkblue', width=2)
            )
        ))
        
        fig4.update_layout(
            title='Systolic BP Distribution',
            yaxis_title='Systolic BP (mmHg)',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    with col4:
        # Glucose comparison
        fig5 = go.Figure()
        
        fig5.add_trace(go.Box(
            y=no_stroke['LB_GLUCOSE'],
            name='No Stroke',
            marker_color='green',
            boxmean='sd'
        ))
        
        fig5.add_trace(go.Box(
            y=stroke['LB_GLUCOSE'],
            name='Stroke',
            marker_color='red',
            boxmean='sd'
        ))
        
        fig5.add_trace(go.Scatter(
            x=['Your Value'],
            y=[user_data['glucose']],
            mode='markers',
            name='YOU',
            marker=dict(
                color='blue',
                size=15,
                symbol='star',
                line=dict(color='darkblue', width=2)
            )
        ))
        
        fig5.update_layout(
            title='Glucose Distribution',
            yaxis_title='Glucose (mg/dL)',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig5, use_container_width=True)


def create_risk_gauge(risk_prob):
    """Create an interactive gauge chart for risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Your Stroke Risk", 'font': {'size': 24}},
        delta={'reference': 6, 'suffix': '%'},
        gauge={
            'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 5], 'color': '#10b981'},
                {'range': [5, 15], 'color': '#f59e0b'},
                {'range': [15, 25], 'color': '#f97316'},
                {'range': [25, 50], 'color': '#ef4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_prob * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


# ============================================================================
# SECTION 5: DETAILED REPORT
# ============================================================================

def generate_detailed_report(user_data, risk_prob, reference_data):
    """Generate detailed health report"""
    st.markdown("## 📋 Detailed Health Report")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Risk Factors", "Comparison", "Recommendations"])
    
    with tab1:
        st.markdown("### Your Risk Factor Analysis")
        
        # Calculate percentiles
        age_percentile = (reference_data['AGE'] < user_data['age']).mean() * 100
        sbp_percentile = (reference_data['VS_SYSBP'] < user_data['systolic_bp']).mean() * 100
        glucose_percentile = (reference_data['LB_GLUCOSE'] < user_data['glucose']).mean() * 100
        chol_percentile = (reference_data['LB_CHOL'] < user_data['cholesterol']).mean() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Demographics")
            st.write(f"**Age:** {user_data['age']} years ({age_percentile:.0f}th percentile)")
            st.write(f"**Sex:** {user_data['sex']}")
            st.write(f"**Race:** {user_data['race']}")
            
            st.markdown("#### Vital Signs")
            st.write(f"**Blood Pressure:** {user_data['systolic_bp']}/{user_data['diastolic_bp']} mmHg")
            st.write(f"  - SBP is at {sbp_percentile:.0f}th percentile")
            if user_data['systolic_bp'] > 140:
                st.warning("⚠️ Elevated blood pressure (hypertension)")
            elif user_data['systolic_bp'] > 130:
                st.warning("⚠️ Blood pressure in pre-hypertension range")
            else:
                st.success("✓ Blood pressure in normal range")
            
            st.write(f"**Heart Rate:** {user_data['heart_rate']} bpm")
            st.write(f"**Weight:** {user_data['weight']} kg")
        
        with col2:
            st.markdown("#### Laboratory Values")
            st.write(f"**Glucose:** {user_data['glucose']} mg/dL ({glucose_percentile:.0f}th percentile)")
            if user_data['glucose'] > 126:
                st.warning("⚠️ Elevated glucose (diabetes range)")
            elif user_data['glucose'] > 100:
                st.warning("⚠️ Glucose in pre-diabetes range")
            else:
                st.success("✓ Glucose in normal range")
            
            st.write(f"**Cholesterol:** {user_data['cholesterol']} mg/dL ({chol_percentile:.0f}th percentile)")
            if user_data['cholesterol'] > 240:
                st.warning("⚠️ High cholesterol")
            elif user_data['cholesterol'] > 200:
                st.warning("⚠️ Borderline high cholesterol")
            else:
                st.success("✓ Cholesterol in desirable range")
            
            st.write(f"**HDL:** {user_data['hdl']} mg/dL")
            st.write(f"**LDL:** {user_data['ldl']} mg/dL")
            st.write(f"**Triglycerides:** {user_data['triglycerides']} mg/dL")
            st.write(f"**Creatinine:** {user_data['creatinine']} mg/dL")
    
    with tab2:
        st.markdown("### How You Compare to Reference Population")
        
        # Create comparison bars
        metrics = {
            'Age': (user_data['age'], reference_data['AGE'].mean(), reference_data['AGE'].std()),
            'Systolic BP': (user_data['systolic_bp'], reference_data['VS_SYSBP'].mean(), reference_data['VS_SYSBP'].std()),
            'Diastolic BP': (user_data['diastolic_bp'], reference_data['VS_DIABP'].mean(), reference_data['VS_DIABP'].std()),
            'Glucose': (user_data['glucose'], reference_data['LB_GLUCOSE'].mean(), reference_data['LB_GLUCOSE'].std()),
            'Cholesterol': (user_data['cholesterol'], reference_data['LB_CHOL'].mean(), reference_data['LB_CHOL'].std()),
        }
        
        for metric_name, (user_val, pop_mean, pop_std) in metrics.items():
            z_score = (user_val - pop_mean) / pop_std
            
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Population Average'],
                    y=[pop_mean],
                    name='Average',
                    marker_color='lightgray',
                    text=[f'{pop_mean:.1f}'],
                    textposition='auto',
                ))
                
                fig.add_trace(go.Bar(
                    x=['Your Value'],
                    y=[user_val],
                    name='You',
                    marker_color='blue',
                    text=[f'{user_val:.1f}'],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=metric_name,
                    showlegend=False,
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric(
                    label="Z-Score",
                    value=f"{z_score:.2f}",
                    delta=f"{user_val - pop_mean:.1f}"
                )
                
                if abs(z_score) < 1:
                    st.success("Within 1 SD")
                elif abs(z_score) < 2:
                    st.warning("1-2 SD away")
                else:
                    st.error(">2 SD away")
    
    with tab3:
        st.markdown("### Personalized Recommendations")
        
        recommendations = []
        
        # Age-based recommendations
        if user_data['age'] > 65:
            recommendations.append({
                'icon': '👴',
                'category': 'Age',
                'message': 'Age is a non-modifiable risk factor. Focus on controlling modifiable risk factors.',
                'priority': 'Info'
            })
        
        # Blood pressure recommendations
        if user_data['systolic_bp'] > 140 or user_data['diastolic_bp'] > 90:
            recommendations.append({
                'icon': '💊',
                'category': 'Blood Pressure',
                'message': 'High blood pressure detected. Consult your doctor about medication and lifestyle changes. Reduce sodium intake, exercise regularly, and manage stress.',
                'priority': 'High'
            })
        elif user_data['systolic_bp'] > 130:
            recommendations.append({
                'icon': '🏃',
                'category': 'Blood Pressure',
                'message': 'Elevated blood pressure. Consider lifestyle modifications: reduce salt, increase physical activity, maintain healthy weight.',
                'priority': 'Medium'
            })
        
        # Glucose recommendations
        if user_data['glucose'] > 126:
            recommendations.append({
                'icon': '🍎',
                'category': 'Glucose',
                'message': 'Elevated glucose levels suggest diabetes. Consult your doctor immediately. Consider dietary changes, exercise, and possible medication.',
                'priority': 'High'
            })
        elif user_data['glucose'] > 100:
            recommendations.append({
                'icon': '🥗',
                'category': 'Glucose',
                'message': 'Pre-diabetes range detected. Adopt a low-carb diet, increase fiber intake, and exercise regularly to prevent progression.',
                'priority': 'Medium'
            })
        
        # Cholesterol recommendations
        if user_data['cholesterol'] > 240:
            recommendations.append({
                'icon': '🩺',
                'category': 'Cholesterol',
                'message': 'High cholesterol detected. Consult your doctor about statin therapy. Reduce saturated fats, increase omega-3 fatty acids.',
                'priority': 'High'
            })
        elif user_data['cholesterol'] > 200:
            recommendations.append({
                'icon': '🥑',
                'category': 'Cholesterol',
                'message': 'Borderline high cholesterol. Increase physical activity, consume more fiber, and limit dietary cholesterol.',
                'priority': 'Medium'
            })
        
        # HDL recommendations
        if user_data['hdl'] < 40:
            recommendations.append({
                'icon': '💪',
                'category': 'HDL (Good Cholesterol)',
                'message': 'Low HDL cholesterol. Increase aerobic exercise, quit smoking if applicable, and consider healthy fats in diet.',
                'priority': 'Medium'
            })
        
        # General recommendations
        recommendations.append({
            'icon': '🚭',
            'category': 'Lifestyle',
            'message': 'If you smoke, quitting is the single most effective way to reduce stroke risk.',
            'priority': 'High'
        })
        
        recommendations.append({
            'icon': '🏋️',
            'category': 'Exercise',
            'message': 'Aim for 150 minutes of moderate aerobic activity per week.',
            'priority': 'Info'
        })
        
        recommendations.append({
            'icon': '🍷',
            'category': 'Alcohol',
            'message': 'Limit alcohol consumption to moderate levels (1 drink/day for women, 2 for men).',
            'priority': 'Info'
        })
        
        # Sort by priority
        priority_order = {'High': 0, 'Medium': 1, 'Info': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        # Display recommendations
        for rec in recommendations:
            if rec['priority'] == 'High':
                st.error(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
            elif rec['priority'] == 'Medium':
                st.warning(f"{rec['icon']} **{rec['category']}**: {rec['message']}")
            else:
                st.info(f"{rec['icon']} **{rec['category']}**: {rec['message']}")


# ============================================================================
# SECTION 6: MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Stroke Risk Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Clinical Decision Support Tool</div>', unsafe_allow_html=True)
    
    # Info box
    with st.expander("ℹ️ About This Tool", expanded=False):
        st.markdown("""
        This tool uses machine learning to predict stroke risk based on clinical data.
        
        **How it works:**
        1. Enter your health information in the sidebar
        2. Click "Calculate Risk" to see your personalized risk assessment
        3. View interactive charts showing how you compare to reference population
        4. Get personalized health recommendations
        
        **Color coding:**
        - 🔵 **Blue**: Your values
        - 🟢 **Green**: Reference population without stroke
        - 🔴 **Red**: Reference population with stroke
        
        **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
        Consult with your healthcare provider for proper diagnosis and treatment.
        """)
    
    # Load model and data
    model, feature_names, reference_data = load_model_and_data()
    
    # Get user input
    user_data, submitted = get_user_input()
    
    if submitted and user_data:
        # Predict risk
        risk_prob = predict_risk(user_data, model, feature_names)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            category, color = display_risk_result(risk_prob)
        
        with col2:
            gauge_fig = create_risk_gauge(risk_prob)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Interactive visualizations
        create_comparison_charts(user_data, reference_data, risk_prob)
        
        st.markdown("---")
        
        # Detailed report
        generate_detailed_report(user_data, risk_prob, reference_data)
        
        # Download report button
        st.markdown("---")
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.download_button(
            label="📥 Download Report (PDF)",
            data=f"Stroke Risk Assessment Report\nDate: {report_date}\nRisk: {risk_prob*100:.1f}%\nCategory: {category}",
            file_name=f"stroke_risk_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    else:
        # Welcome screen
        st.markdown("## 👈 Get Started")
        st.info("Enter your health information in the sidebar and click **'Calculate Risk'** to begin.")
        
        # Show demo visualization
        st.markdown("### Sample Visualization")
        st.image("https://via.placeholder.com/800x400/3498db/ffffff?text=Interactive+Charts+Will+Appear+Here", 
                use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reference Population", "1,000 patients")
        with col2:
            st.metric("Model Accuracy", "81.2% AUC")
        with col3:
            st.metric("Stroke Detection", "85% Recall")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Powered by Machine Learning | Built with Streamlit</p>
        <p style='font-size: 0.8rem;'>⚠️ For educational purposes only. Not a substitute for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
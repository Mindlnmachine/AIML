import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# Page config
st.set_page_config(page_title="Real-time Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Real-time Expense Fraud Detection System")
st.markdown("**Data Science Approach: Outlier Detection for Fraud Analysis**")

# Generate demo dataset
@st.cache_data
def generate_demo_data(n_samples=10000):
    np.random.seed(42)
    
    # Normal employees (90%)
    normal_size = int(n_samples * 0.9)
    normal_data = {
        'EmployeeID': range(1, normal_size + 1),
        'Name': [f'Employee_{i}' for i in range(1, normal_size + 1)],
        'Department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], normal_size),
        'Allowance': np.random.normal(5000, 1000, normal_size),
        'ExpenseAmount': np.random.normal(4500, 800, normal_size),
        'WorkLogs': np.random.normal(160, 20, normal_size),
        'ProjectCount': np.random.poisson(3, normal_size),
        'OvertimeHours': np.random.exponential(10, normal_size)
    }
    
    # Fraudulent employees (10%) - outliers
    fraud_size = n_samples - normal_size
    fraud_data = {
        'EmployeeID': range(normal_size + 1, n_samples + 1),
        'Name': [f'Employee_{i}' for i in range(normal_size + 1, n_samples + 1)],
        'Department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], fraud_size),
        'Allowance': np.random.normal(5000, 1000, fraud_size),
        'ExpenseAmount': np.random.normal(7500, 1500, fraud_size),  # Higher expenses
        'WorkLogs': np.random.normal(80, 30, fraud_size),  # Lower work logs
        'ProjectCount': np.random.poisson(1, fraud_size),  # Fewer projects
        'OvertimeHours': np.random.exponential(2, fraud_size)  # Less overtime
    }
    
    # Combine data
    df = pd.DataFrame({key: np.concatenate([normal_data[key], fraud_data[key]]) 
                      for key in normal_data.keys()})
    
    # Ensure positive values
    df['Allowance'] = np.abs(df['Allowance'])
    df['ExpenseAmount'] = np.abs(df['ExpenseAmount'])
    df['WorkLogs'] = np.abs(df['WorkLogs'])
    df['OvertimeHours'] = np.abs(df['OvertimeHours'])
    
    # Add some derived features
    df['ExpenseRatio'] = df['ExpenseAmount'] / df['Allowance']
    df['ProductivityScore'] = (df['WorkLogs'] * df['ProjectCount']) / (df['OvertimeHours'] + 1)
    
    return df

# Load or generate data
df = generate_demo_data()

# Sidebar controls
st.sidebar.header("🔧 Detection Controls")

# Real-time simulation
if st.sidebar.button("🔄 Run Real-time Fraud Detection", type="primary"):
    st.sidebar.success("✅ Analysis Running...")
    
    # Feature selection for outlier detection
    feature_cols = ['ExpenseAmount', 'WorkLogs', 'ExpenseRatio', 'ProductivityScore', 'OvertimeHours']
    X = df[feature_cols].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Multiple outlier detection methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Isolation Forest Detection")
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['IsoForest_Outlier'] = iso_forest.fit_predict(X_scaled)
        df['IsoForest_Fraud'] = df['IsoForest_Outlier'] == -1
        
        iso_fraud_count = df['IsoForest_Fraud'].sum()
        st.metric("Fraud Cases Detected", iso_fraud_count, delta=f"{iso_fraud_count/len(df)*100:.1f}%")
    
    with col2:
        st.subheader("🎯 DBSCAN Clustering")
        dbscan = DBSCAN(eps=0.5, min_samples=50)
        clusters = dbscan.fit_predict(X_scaled)
        df['DBSCAN_Outlier'] = clusters == -1
        
        dbscan_fraud_count = df['DBSCAN_Outlier'].sum()
        st.metric("Outliers Found", dbscan_fraud_count, delta=f"{dbscan_fraud_count/len(df)*100:.1f}%")
    
    # Statistical outlier detection
    st.subheader("📊 Statistical Outlier Analysis")
    
    # Z-score method
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, axis=0))
    df['ZScore_Outlier'] = (z_scores > 3).any(axis=1)
    
    # IQR method
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    df['IQR_Outlier'] = (detect_outliers_iqr(df, 'ExpenseAmount') | 
                        detect_outliers_iqr(df, 'ExpenseRatio'))
    
    # Combine all methods for final fraud prediction
    df['Final_Fraud'] = (df['IsoForest_Fraud'] | df['DBSCAN_Outlier'] | 
                        df['ZScore_Outlier'] | df['IQR_Outlier'])
    
    # Display results
    total_fraud = df['Final_Fraud'].sum()
    
    # if total_fraud > 0:
    #     st.markdown(f"""
    #     <div class="fraud-alert">
    #         <h3>⚠️ FRAUD ALERT!</h3>
    #         <p><strong>{total_fraud}</strong> potentially fraudulent transactions detected out of {len(df)} records</p>
    #         <p>Fraud Rate: <strong>{total_fraud/len(df)*100:.2f}%</strong></p>
    #     </div>
    #     """, unsafe_allow_html=True)

    if total_fraud > 0:
        st.markdown("""
        <style>
        .fraud-alert {
        background-color: #ffcccc;          /* Light red background */
        border: 2px solid #ff4d4d;          /* Red border */
        border-radius: 10px;                /* Rounded corners */
        padding: 15px;                      /* Inner spacing */
        color: #660000;                     /* Dark red text */
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
        box-shadow: 0 0 10px rgba(255, 0, 0, 0.2); /* Subtle shadow */
        }
        .fraud-alert h3 {
            color: #b30000;                     /* Slightly brighter for title */
            margin-top: 0;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fraud-alert">
            <h3>⚠️ FRAUD ALERT!</h3>
            <p><strong>{total_fraud}</strong> potentially fraudulent transactions detected out of {len(df)} records</p>
            <p>Fraud Rate: <strong>{total_fraud/len(df)*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-alert">
            <h3>✅ ALL CLEAR</h3>
            <p>No fraudulent patterns detected in the current dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed fraud analysis
    if total_fraud > 0:
        st.subheader("🔍 Detailed Fraud Analysis")
        
        fraud_df = df[df['Final_Fraud']].copy()
        
        # Show top fraudulent cases
        st.write("**Top 20 Fraudulent Cases:**")
        display_cols = ['EmployeeID', 'Name', 'Department', 'ExpenseAmount', 'Allowance', 
                       'ExpenseRatio', 'WorkLogs', 'ProductivityScore']
        st.dataframe(fraud_df[display_cols].head(20), use_container_width=True)
        
        # Simple and Clear Visualizations
        st.subheader("📈 Easy-to-Understand Fraud Analysis")
        
        # Get data for plotting
        normal_data = df[~df['Final_Fraud']]
        fraud_data = df[df['Final_Fraud']]
        
        # Row 1: Simple comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Simple Pie Chart - Fraud vs Normal
            st.write("**🥧 Overall Fraud Distribution**")
            fraud_counts = df['Final_Fraud'].value_counts()
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors = ['#28a745', '#dc3545']  # Green for Normal, Red for Fraud
            labels = ['Normal Employees', 'Fraudulent Employees']
            sizes = [fraud_counts[False], fraud_counts[True]]
            
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 12})
            ax1.set_title('Fraud Detection Results', fontsize=14, fontweight='bold')
            st.pyplot(fig1)
        
        with col2:
            # Simple Bar Chart - Department wise fraud
            st.write("**📊 Fraud Cases by Department**")
            dept_fraud = df.groupby('Department')['Final_Fraud'].sum().sort_values(ascending=True)
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            bars = ax2.barh(dept_fraud.index, dept_fraud.values, color='#ff6b6b')
            ax2.set_xlabel('Number of Fraud Cases', fontsize=12)
            ax2.set_title('Which Department has Most Fraud?', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                            f'{int(width)}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # Row 2: Simple comparison charts  
        col3, col4 = st.columns(2)
        
        with col3:
            # Simple comparison - Average Expense Amount
            st.write("**💰 Average Expense: Normal vs Fraud**")
            avg_normal = normal_data['ExpenseAmount'].mean()
            avg_fraud = fraud_data['ExpenseAmount'].mean()
            
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            categories = ['Normal\nEmployees', 'Fraudulent\nEmployees']
            values = [avg_normal, avg_fraud]
            colors = ['#28a745', '#dc3545']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.8)
            ax3.set_ylabel('Average Expense Amount (₹)', fontsize=12)
            ax3.set_title('Who Spends More Money?', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'₹{value:.0f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col4:
            # Simple comparison - Average Work Hours
            st.write("**⏰ Average Work Hours: Normal vs Fraud**")
            avg_work_normal = normal_data['WorkLogs'].mean()
            avg_work_fraud = fraud_data['WorkLogs'].mean()
            
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            categories = ['Normal\nEmployees', 'Fraudulent\nEmployees']
            values = [avg_work_normal, avg_work_fraud]
            colors = ['#28a745', '#dc3545']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.8)
            ax4.set_ylabel('Average Work Hours', fontsize=12)
            ax4.set_title('Who Works Less Hours?', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{value:.0f}h', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig4)
        
        # Additional Simple Analysis
        st.subheader("📊 Key Insights")
        
        # Create insights in simple text format
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💰 Average Fraud Amount", 
                f"₹{fraud_data['ExpenseAmount'].mean():.0f}",
                delta=f"₹{fraud_data['ExpenseAmount'].mean() - normal_data['ExpenseAmount'].mean():.0f} higher"
            )
        
        with col2:
            st.metric(
                "⏰ Average Fraud Work Hours", 
                f"{fraud_data['WorkLogs'].mean():.0f}h",
                delta=f"{fraud_data['WorkLogs'].mean() - normal_data['WorkLogs'].mean():.0f}h lower"
            )
        
        with col3:
            worst_dept = df.groupby('Department')['Final_Fraud'].sum().idxmax()
            worst_dept_count = df.groupby('Department')['Final_Fraud'].sum().max()
            st.metric(
                "🏢 Most Fraud Department", 
                worst_dept,
                delta=f"{worst_dept_count} cases"
            )
        
        # Simple Summary Box
        st.success(f"""
        **🔍 Quick Summary:**
        - **{total_fraud}** employees are doing fraud out of **{len(df)}** total
        - Fraudulent employees spend **₹{fraud_data['ExpenseAmount'].mean() - normal_data['ExpenseAmount'].mean():.0f} more** on average
        - They work **{normal_data['WorkLogs'].mean() - fraud_data['WorkLogs'].mean():.0f} hours less** on average  
        - **{worst_dept}** department has the most fraud cases
        """)
        
        # Real-time monitoring simulation
        st.subheader("🔴 Real-time Monitoring")
        
        # Simulate incoming transactions
        if st.button("Simulate New Transaction"):
            new_transaction = df.sample(1).iloc[0]
            
            st.write("**New Transaction Received:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Employee ID", new_transaction['EmployeeID'])
                st.metric("Department", new_transaction['Department'])
            
            with col2:
                st.metric("Expense Amount", f"₹{new_transaction['ExpenseAmount']:.2f}")
                st.metric("Allowance", f"₹{new_transaction['Allowance']:.2f}")
            
            with col3:
                st.metric("Expense Ratio", f"{new_transaction['ExpenseRatio']:.2f}")
                st.metric("Work Logs", f"{new_transaction['WorkLogs']:.0f}")
            
            # Real-time prediction
            transaction_features = new_transaction[feature_cols].values.reshape(1, -1)
            transaction_scaled = scaler.transform(transaction_features)
            
            fraud_score = iso_forest.decision_function(transaction_scaled)[0]
            is_fraud = iso_forest.predict(transaction_scaled)[0] == -1
            
            if is_fraud:
                st.error(f"🚨 **FRAUD DETECTED!** Risk Score: {abs(fraud_score):.3f}")
                st.warning("⚠️ Transaction flagged for manual review")
            else:
                st.success(f"✅ **Transaction Normal** Risk Score: {abs(fraud_score):.3f}")

# Show dataset info
st.sidebar.markdown("---")
st.sidebar.write("📊 **Dataset Info:**")
st.sidebar.write(f"Total Employees: {len(df):,}")
st.sidebar.write(f"Departments: {df['Department'].nunique()}")

# File uploader
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Custom Dataset", type=['csv'])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.subheader("📁 Uploaded Dataset Analysis")
        st.write(f"Dataset shape: {uploaded_df.shape}")
        st.dataframe(uploaded_df.head())
        
        if st.button("Analyze Uploaded Data"):
            st.info("Run the same analysis on your uploaded dataset...")
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Built with:** Streamlit, Scikit-learn, Matplotlib, Seaborn | **Real-time Fraud Detection System**")
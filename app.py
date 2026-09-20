import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for a wider, cleaner layout
st.set_page_config(page_title="Macro Quant Platform", page_icon="🌍", layout="wide")

# Use seaborn whitegrid for more academic/cleaner charts
sns.set_style("whitegrid")

# Main Header
st.title("🌍 Macroeconomic & Exchange Rate Quantitative Platform")
st.markdown("""
*An interactive econometric tool for validating **Purchasing Power Parity (PPP)** and the **Monetary Approach to the Exchange Rate (MAER)**.*
""")
st.divider()

# Sidebar for Data Upload
st.sidebar.header("📁 1. Data Import")
uploaded_file = st.sidebar.file_uploader("Upload Macroeconomic CSV Data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully!")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Define modern Tabs
    tab1, tab2 = st.tabs(["📈 Purchasing Power Parity (PPP)", "🏦 Monetary Approach (MAER)"])
    
    # ==========================================
    # TAB 1: PPP Test
    # ==========================================
    with tab1:
        st.markdown("### Model Specification")
        st.latex(r"\Delta\ln(E) = \alpha + \beta (\Delta\ln(CPI_A) - \Delta\ln(CPI_B)) + \epsilon")
        st.caption("**Null Hypothesis ($H_0$)**: $\beta = 1$ (Inflation differential is fully reflected in the exchange rate)")
        st.divider()

        st.markdown("### Variable Selection")
        c1, c2, c3 = st.columns(3)
        with c1: e_col = st.selectbox("Exchange Rate (E)", cols, key="ppp_e")
        with c2: cpi_a_col = st.selectbox("Country A CPI", cols, key="ppp_cpi_a")
        with c3: cpi_b_col = st.selectbox("Country B CPI", cols, key="ppp_cpi_b")
            
        if st.button("Run OLS Regression", type="primary", key="btn_ppp"):
            # Data Processing
            temp_df = pd.DataFrame()
            temp_df['dln_E'] = np.log(df[e_col]).diff()
            temp_df['pi_A'] = np.log(df[cpi_a_col]).diff()
            temp_df['pi_B'] = np.log(df[cpi_b_col]).diff()
            temp_df['inf_diff'] = temp_df['pi_A'] - temp_df['pi_B']
            
            model_data = temp_df.dropna()
            Y = model_data['dln_E']
            X = sm.add_constant(model_data['inf_diff'])
            model = sm.OLS(Y, X).fit()
            
            st.divider()
            
            # Split results and chart for a cleaner layout
            res_col, chart_col = st.columns([1, 1.2])
            
            with res_col:
                st.markdown("### 📊 Regression Summary")
                st.text(model.summary())
                
                beta = model.params['inf_diff']
                p_val = model.pvalues['inf_diff']
                
                st.markdown("#### Analytical Conclusion")
                if p_val < 0.05:
                    st.success(f"**Statistically Significant** (p-value = {p_val:.4f}). \n\nEstimated $\\beta$ = **{beta:.4f}**.")
                    t_stat = (beta - 1) / model.bse['inf_diff']
                    st.info(f"**Deep Dive**: The t-statistic for $H_0: \\beta=1$ is **{t_stat:.3f}**.")
                else:
                    st.warning(f"**Not Significant** (p-value = {p_val:.4f}). \n\nThe data does not robustly support PPP.")

            with chart_col:
                st.markdown("### 📉 Data Fit Visualization")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.regplot(x=model_data['inf_diff'], y=model_data['dln_E'], ax=ax, 
                            scatter_kws={'alpha':0.6, 'color':'#2E86C1'}, line_kws={'color':'#E74C3C'})
                ax.set_xlabel("Inflation Log-Differential $(\Delta\ln CPI_A - \Delta\ln CPI_B)$")
                ax.set_ylabel("Exchange Rate Log-Return $(\Delta\ln E)$")
                ax.set_title("PPP OLS Regression Fit")
                plt.tight_layout()
                st.pyplot(fig)

    # ==========================================
    # TAB 2: MAER Test
    # ==========================================
    with tab2:
        st.markdown("### Model Specification")
        st.latex(r"\hat{E} = \hat{M}_A - \hat{M}_B - \Delta R_B + \hat{Y}_B + \Delta R_A - \hat{Y}_A")
        
        st.info("💡 **Data Standard**: Ensure Output (Y) uses **Real GDP** (absolute volume). Avoid using 'Change in Inventories' to prevent percentage distortion.", icon="ℹ️")
        rate_scale = st.checkbox("Policy Rates are in percentages (e.g., '5.0' for 5%). Auto-divide by 100.", value=True)
        st.divider()

        st.markdown("### Variable Selection")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Country A")
            m_a_col = st.selectbox("Money Supply (M)", cols, key="ma")
            y_a_col = st.selectbox("Real GDP (Y)", cols, key="ya")
            r_a_col = st.selectbox("Policy Rate (R)", cols, key="ra")
        with c2:
            st.markdown("#### Country B")
            m_b_col = st.selectbox("Money Supply (M)", cols, key="mb")
            y_b_col = st.selectbox("Real GDP (Y)", cols, key="yb")
            r_b_col = st.selectbox("Policy Rate (R)", cols, key="rb")
        with c3:
            st.markdown("#### Exchange Rate")
            e_maer_col = st.selectbox("Exchange Rate (E)", cols, key="e_maer")
            
        if st.button("Run MAER Simulation", type="primary", key="btn_maer"):
            st.divider()
            
            # Computation
            m_df = pd.DataFrame()
            m_df['M_A_hat'] = df[m_a_col].pct_change()
            m_df['M_B_hat'] = df[m_b_col].pct_change()
            m_df['Y_A_hat'] = df[y_a_col].pct_change()
            m_df['Y_B_hat'] = df[y_b_col].pct_change()
            
            scale_factor = 100.0 if rate_scale else 1.0
            m_df['R_A_diff'] = df[r_a_col].diff() / scale_factor
            m_df['R_B_diff'] = df[r_b_col].diff() / scale_factor
            m_df['Actual_E_hat'] = np.log(df[e_maer_col]).diff()
            
            m_df['MAER_E_hat'] = (m_df['M_A_hat'] - m_df['M_B_hat'] - m_df['R_B_diff'] + 
                                  m_df['Y_B_hat'] + m_df['R_A_diff'] - m_df['Y_A_hat'])
            m_df = m_df.dropna()
            
            st.markdown("### 🧮 Results & Visualization")
            
            metric_col1, metric_col2, _ = st.columns([1, 1, 2])
            metric_col1.metric(label="Mean Actual Change ($\hat{E}$)", value=f"{m_df['Actual_E_hat'].mean():.6f}")
            metric_col2.metric(label="Mean Theoretical Change (MAER)", value=f"{m_df['MAER_E_hat'].mean():.6f}")
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(m_df.index, m_df['Actual_E_hat'], label="Actual Exchange Rate Change", marker='o', color='#2E86C1')
            ax2.plot(m_df.index, m_df['MAER_E_hat'], label="MAER Theoretical Change", marker='x', linestyle='--', color='#E74C3C')
            ax2.set_xlabel("Time Index")
            ax2.set_ylabel("Rate of Change (Decimal)")
            ax2.set_title("Actual vs. MAER Theoretical Exchange Rate Change")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
            
            with st.expander("🕵️‍♂️ View Diagnostic Data Table"):
                st.dataframe(m_df.style.format("{:.6f}"))

else:
    st.info("👈 Please upload your macroeconomic dataset (CSV format) from the sidebar to begin.")

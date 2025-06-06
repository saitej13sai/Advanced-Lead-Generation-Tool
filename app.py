import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
import numpy as np
from urllib.error import HTTPError

# Function to scrape mock company data (simulating Crunchbase or similar)
def scrape_leads(url="https://example.com"):
    try:
        # Mock response for demo purposes (replace with real scraping if permitted)
        mock_data = [
            {'name': 'TechTrend Inc', 'industry': 'SaaS', 'size': 150, 'revenue': 5000000, 'website': 'techtrend.com'},
            {'name': 'GrowEasy Ltd', 'industry': 'SaaS', 'size': 80, 'revenue': 3000000, 'website': 'groweasy.com'},
            {'name': 'SteelWorks Co', 'industry': 'Manufacturing', 'size': 200, 'revenue': 10000000, 'website': 'steelworks.com'},
            {'name': 'CloudPeak', 'industry': 'SaaS', 'size': 50, 'revenue': 2000000, 'website': 'cloudpeak.com'},
            {'name': 'BuildFast', 'industry': 'Construction', 'size': 300, 'revenue': 15000000, 'website': 'buildfast.com'}
        ]
        df = pd.DataFrame(mock_data)
        
        # Basic data cleaning
        df['size'] = df['size'].fillna(0).astype(int)
        df['revenue'] = df['revenue'].fillna(0).astype(float)
        df['industry'] = df['industry'].fillna('Unknown')
        return df
    except HTTPError as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to score leads using a simple ML model
def score_leads(df):
    if df.empty:
        return df
    
    # Feature engineering
    df['is_saas'] = (df['industry'] == 'SaaS').astype(int)
    df['size_scaled'] = df['size'] / df['size'].max()  # Normalize size
    df['revenue_scaled'] = df['revenue'] / df['revenue'].max()  # Normalize revenue
    
    # Mock target variable (1 for high-value lead, 0 for low-value)
    df['target'] = ((df['is_saas'] == 1) & (df['size'] > 50) & (df['revenue'] > 2000000)).astype(int)
    
    # Prepare features for model
    X = df[['size_scaled', 'revenue_scaled', 'is_saas']]
    y = df['target']
    
    # Train a simple logistic regression model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Predict lead scores (probability of being high-value)
    df['score'] = model.predict_proba(X)[:, 1] * 100  # Scale to 0-100
    return df[['name', 'industry', 'size', 'revenue', 'website', 'score']].sort_values('score', ascending=False)

# Streamlit UI
def main():
    st.set_page_config(page_title="Caprae Lead Scoring Tool", layout="wide")
    
    # Custom CSS for polished design
    st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    .stButton>button { background-color: #1e88e5; color: white; border-radius: 5px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #1e88e5; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("Caprae AI-Powered Lead Scoring Tool")
    st.markdown("""
    This tool enhances lead generation by scoring companies based on their potential value for sales outreach.
    It prioritizes SaaS companies with high revenue and employee count, aligning with Caprae's SaaS/MaaS model.
    Built for Caprae Capital Partners' AI-Readiness Challenge.
    """)
    
    # Scrape data
    st.subheader("Lead Data")
    leads_df = scrape_leads()
    
    if not leads_df.empty:
        # Score leads
        scored_leads = score_leads(leads_df)
        
        # Display results
        st.subheader("Scored Leads")
        st.markdown("Leads are scored from 0-100 based on industry, size, and revenue. Higher scores indicate higher priority.")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            industry_filter = st.multiselect("Filter by Industry", options=scored_leads['industry'].unique(), default=scored_leads['industry'].unique())
        with col2:
            min_score = st.slider("Minimum Score", 0, 100, 50)
        
        # Apply filters
        filtered_leads = scored_leads[scored_leads['industry'].isin(industry_filter) & (scored_leads['score'] >= min_score)]
        
        # Display table
        st.dataframe(
            filtered_leads.style.format({
                'size': '{:,.0f}',
                'revenue': '${:,.0f}',
                'score': '{:.1f}'
            }).set_properties(**{'text-align': 'left'}),
            use_container_width=True
        )
        
        # Chart: Lead score distribution by industry
        st.subheader("Lead Score Distribution by Industry")
        industry_scores = filtered_leads.groupby('industry')['score'].mean().reset_index()
        st.bar_chart(industry_scores.set_index('industry')['score'])
        
        # Download button for CSV export
        csv = filtered_leads.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Leads as CSV",
            data=csv,
            file_name="scored_leads.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data available. Please check the data source or try again later.")

if __name__ == "__main__":
    main()

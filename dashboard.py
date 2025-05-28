import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from datetime import datetime
from pathlib import Path
import os
import numpy as np

# Path configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

def load_assets():
    """Load CSS and JavaScript files"""
    try:
        # Load CSS
        css_file = STATIC_DIR / "styles.css"
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
        # Load JavaScript
        js_file = STATIC_DIR / "scripts.js"
        with open(js_file) as f:
            st.components.v1.html(
                f"""<script>{f.read()}</script>""",
                height=0,
                width=0,
            )
    except Exception as e:
        st.error(f"Error loading static assets: {e}")

# Page configuration
st.set_page_config(
    page_title="Wowowowow Sales Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Load assets
load_assets()

# Database connection
@st.cache_resource
def get_connection():
    return create_engine(
        "postgresql://user:szrVeYFcCzLxNRLlxxPtX6w4fWqM1ulr@dpg-d0oun3muk2gs73945erg-a.singapore-postgres.render.com/salesdashboard_of09"
    )

@st.cache_data(ttl=3600)
def load_data():
    try:
        engine = get_connection()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM dashboard_data"))
            df = pd.DataFrame(result.mappings().all())

        # Convert to datetime
        date_cols = ['order_date', 'ship_date', 'due_date', 'birth_date',
                    'customer_creation_date', 'product_start_date', 'product_end_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')

        # Normalize country names
        df['CNTRY'] = df['CNTRY'].str.upper()
        country_map = {
            'US': 'UNITED STATES',
            'USA': 'UNITED STATES',
            'UNITED STATES': 'UNITED STATES',
            'DE': 'GERMANY',
            'GERMENY': 'GERMANY',
            'GERMANY': 'GERMANY'
        }
        df['CNTRY'] = df['CNTRY'].map(country_map).fillna(df['CNTRY'])

        # Normalize gender values
        gender_map = {
            'MALE': 'Male',
            'FEMALE': 'Female'
        }
        df['GEN'] = df['GEN'].map(lambda x: gender_map.get(str(x).upper(), x))

        # Derived columns
        df['customer_age'] = ((datetime.now() - df['birth_date']).dt.days / 365).astype(int)
        df['Revenue'] = df['sls_quantity'] * df['sls_price']
        df['Cost'] = df['sls_quantity'] * df['prd_cost']
        df['Profit'] = df['Revenue'] - df['Cost']
        df['Margin'] = (df['Profit'] / df['Revenue'].replace(0, pd.NA)) * 100
        df['order_year_month'] = df['order_date'].dt.to_period('M')
        df['order_to_ship_days'] = (df['ship_date'] - df['order_date']).dt.days
        df['ship_delay'] = (df['due_date'] - df['ship_date']).dt.days

        # Keep datetime columns for resampling, create separate date columns for display
        df['order_date_display'] = df['order_date'].dt.date
        df['ship_date_display'] = df['ship_date'].dt.date
        df['due_date_display'] = df['due_date'].dt.date

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Custom title with animation
st.markdown("""
    <h1 style="text-align: center; padding: 1rem 0; position: sticky; top: 0; 
               background: linear-gradient(180deg, #f8f9fa 90%, transparent); 
               z-index: 1000; transition: opacity 0.3s ease;">
        WowowowoW Sales Dashboard
        <div style="height: 2px; background: linear-gradient(90deg, #6c5ce7 0%, #a66efa 100%); 
                    width: 60px; margin: 0.5rem auto;"></div>
    </h1>
""", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    # Use display dates for filters but convert back to datetime for filtering
    min_date = df['order_date'].dt.date.min() if not df.empty else datetime.now().date()
    max_date = df['order_date'].dt.date.max() if not df.empty else datetime.now().date()
    
    start_date = st.date_input("Start date", min_date)
    end_date = st.date_input("End date", max_date)

    available_countries = sorted(df['CNTRY'].dropna().unique()) if not df.empty else []
    selected_countries = st.multiselect("Countries", options=available_countries)

    available_categories = sorted(df['CAT'].dropna().unique()) if not df.empty else []
    selected_categories = st.multiselect("Categories", options=available_categories)

# Filter logic - use datetime columns for filtering
if not df.empty:
    mask = (
        (df['order_date'].dt.date >= start_date) &
        (df['order_date'].dt.date <= end_date) &
        (df['CNTRY'].isin(selected_countries if selected_countries else df['CNTRY'].unique())) &
        (df['CAT'].isin(selected_categories if selected_categories else df['CAT'].unique()))
    )
    filtered_df = df[mask]
else:
    filtered_df = df

# Key metrics
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"${filtered_df['Revenue'].sum():,.0f}")
    with col2:
        avg_margin = filtered_df['Margin'].mean()
        st.metric("Average Margin", f"{avg_margin:.1f}%" if pd.notna(avg_margin) else "N/A")
    with col3:
        st.metric("Active Customers", filtered_df['CID'].nunique())
    with col4:
        avg_order_value = filtered_df['Revenue'].mean()
        st.metric("Avg Order Value", f"${avg_order_value:.0f}" if pd.notna(avg_order_value) else "N/A")
else:
    st.warning("No data available for the selected filters.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sales Analysis", "Customer Insights", "Product Analytics", "Operations"])

with tab1:
    if not filtered_df.empty:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Fixed resample method - use set_index approach
                monthly_rev = filtered_df.set_index('order_date').resample('M')['Revenue'].sum().reset_index()
                fig = px.line(monthly_rev, x='order_date', y='Revenue', 
                              title="Monthly Revenue Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating monthly revenue chart: {e}")
        
        with col2:
            try:
                country_rev = filtered_df.groupby('CNTRY')['Revenue'].sum().nlargest(10).reset_index()
                fig = px.bar(country_rev, x='CNTRY', y='Revenue', 
                             title="Top 6 Countries by Revenue",
                             color='Revenue', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating country revenue chart: {e}")
        
        try:
            # Fixed category trend - use groupby with pd.Grouper
            category_trend = filtered_df.groupby([pd.Grouper(key='order_date', freq='M'), 'CAT'])['Revenue'].sum().reset_index()
            fig = px.area(category_trend, x='order_date', y='Revenue', 
                          color='CAT', title="Revenue Trend by Category")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating category trend chart: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for Sales Analysis.")

with tab2:
    if not filtered_df.empty:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        # Enhanced Customer Analysis Section (REPLACEMENT)
        try:
            # Calculate comprehensive customer metrics
            customer_metrics = filtered_df.groupby('CID').agg({
                'order_date': ['min', 'max', 'count'],
                'Revenue': ['sum', 'mean', 'std'],
                'Profit': 'sum',
                'sls_ord_num': 'nunique',
                'CAT': 'nunique',
                'CNTRY': 'first',
                'GEN': 'first',
                'customer_age': 'first'
            }).reset_index()
            
            # Flatten column names
            customer_metrics.columns = ['CID', 'first_order', 'last_order', 'total_orders', 
                                      'total_revenue', 'avg_order_value', 'revenue_std',
                                      'total_profit', 'unique_orders', 'categories_bought',
                                      'country', 'gender', 'age']
            
            # Calculate additional metrics
            customer_metrics['customer_lifespan_days'] = (customer_metrics['last_order'] - customer_metrics['first_order']).dt.days
            customer_metrics['recency_days'] = (datetime.now() - customer_metrics['last_order']).dt.days
            customer_metrics['purchase_frequency'] = customer_metrics['total_orders'] / (customer_metrics['customer_lifespan_days'] + 1)
            customer_metrics['profit_margin'] = (customer_metrics['total_profit'] / customer_metrics['total_revenue']) * 100
            
            # Create customer segments based on RFM analysis
            customer_metrics['recency_score'] = pd.qcut(customer_metrics['recency_days'], 5, labels=[5,4,3,2,1])
            customer_metrics['frequency_score'] = pd.qcut(customer_metrics['total_orders'].rank(method='first'), 5, labels=[1,2,3,4,5])
            customer_metrics['monetary_score'] = pd.qcut(customer_metrics['total_revenue'], 5, labels=[1,2,3,4,5])
            
            # Convert to numeric for calculation
            customer_metrics['recency_score'] = pd.to_numeric(customer_metrics['recency_score'])
            customer_metrics['frequency_score'] = pd.to_numeric(customer_metrics['frequency_score'])
            customer_metrics['monetary_score'] = pd.to_numeric(customer_metrics['monetary_score'])
            
            customer_metrics['rfm_score'] = (customer_metrics['recency_score'] + 
                                           customer_metrics['frequency_score'] + 
                                           customer_metrics['monetary_score']) / 3
            
            # Define customer segments
            def get_customer_segment(row):
                if row['rfm_score'] >= 4.5:
                    return 'Champions'
                elif row['rfm_score'] >= 4.0:
                    return 'Loyal Customers'
                elif row['rfm_score'] >= 3.5:
                    return 'Potential Loyalists'
                elif row['rfm_score'] >= 3.0:
                    return 'New Customers'
                elif row['rfm_score'] >= 2.5:
                    return 'Promising'
                elif row['rfm_score'] >= 2.0:
                    return 'Need Attention'
                elif row['rfm_score'] >= 1.5:
                    return 'About to Sleep'
                else:
                    return 'At Risk'
            
            customer_metrics['segment'] = customer_metrics.apply(get_customer_segment, axis=1)
            
            # Clean and normalize gender values in customer_metrics
            customer_metrics['gender'] = (
                customer_metrics['gender']
                .astype(str)
                .str.strip()
                .str.upper()
                .map({'MALE': 'Male', 'FEMALE': 'Female'})
                .fillna('Other')
            )
            # Optional: print unique values for debugging
            print("Unique gender values:", customer_metrics['gender'].unique())
            
            # 1. Key Customer Insights (moved to top)
            st.subheader("Key Customer Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                champions_count = len(customer_metrics[customer_metrics['segment'] == 'Champions'])
                champions_revenue = customer_metrics[customer_metrics['segment'] == 'Champions']['total_revenue'].sum()
                st.metric(
                    "Champions", 
                    f"{champions_count}",
                    f"${champions_revenue:,.0f} Revenue"
                )
            
            with col2:
                at_risk_count = len(customer_metrics[customer_metrics['segment'] == 'At Risk'])
                at_risk_revenue = customer_metrics[customer_metrics['segment'] == 'At Risk']['total_revenue'].sum()
                st.metric(
                    "At Risk", 
                    f"{at_risk_count}",
                    f"${at_risk_revenue:,.0f} at Stake"
                )
            
            with col3:
                avg_customer_lifespan = customer_metrics['customer_lifespan_days'].mean()
                st.metric(
                    "Avg Lifespan", 
                    f"{avg_customer_lifespan:.0f} days",
                    f"{avg_customer_lifespan/30:.1f} months"
                )
            
            with col4:
                repeat_customers = len(customer_metrics[customer_metrics['total_orders'] > 1])
                repeat_rate = (repeat_customers / len(customer_metrics)) * 100
                st.metric(
                    "Repeat Rate", 
                    f"{repeat_rate:.1f}%",
                    f"{repeat_customers} customers"
                )
            
            # 2. Customer Value Distribution Analysis
            st.subheader("Advanced Customer Segmentation Analysis")
            
            # Filter to only Male and Female for plotting
            filtered_customer_metrics = customer_metrics[customer_metrics['gender'].isin(['Male', 'Female'])]
            # Customer Value Distribution by Segment
            fig_violin = px.violin(
                customer_metrics, 
                x='segment', 
                y='total_revenue',
                box=True,
                title="Customer Value Distribution by Segment",
                color='segment',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_violin.update_xaxes(tickangle=45)
            fig_violin.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # Age vs Total Revenue Frequency Polygon by Gender
            bin_width = 5
            filtered_customer_metrics['age_bin'] = (filtered_customer_metrics['age'] // bin_width) * bin_width
            # Group by age_bin and gender, sum total revenue
            revenue_df = (
                filtered_customer_metrics
                .groupby(['age_bin', 'gender'])['total_revenue']
                .sum()
                .reset_index()
            )
            fig_revenue_poly = px.line(
                revenue_df,
                x='age_bin',
                y='total_revenue',
                color='gender',
                markers=True,
                title="Total Revenue by Age and Gender",
                labels={'age_bin': 'Age', 'total_revenue': 'Total Revenue'}
            )
            fig_revenue_poly.update_traces(mode='lines+markers')
            st.plotly_chart(fig_revenue_poly, use_container_width=True, key="age_revenue_polygon")
            
            # 3. Customer Segment Performance Dashboard
            st.subheader("Segment Performance Metrics")
            
            # Calculate segment statistics
            segment_stats = customer_metrics.groupby('segment').agg({
                'CID': 'count',
                'total_revenue': ['mean', 'sum'],
                'avg_order_value': 'mean',
                'total_orders': 'mean',
                'recency_days': 'mean',
                'profit_margin': 'mean'
            }).round(2)
            
            # Flatten column names
            segment_stats.columns = ['customer_count', 'avg_revenue', 'total_segment_revenue', 
                                   'avg_order_value', 'avg_orders', 'avg_recency', 'avg_margin']
            segment_stats = segment_stats.reset_index()
            
            # Create multi-metric visualization
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Revenue per Customer', 'Average Order Value', 
                              'Purchase Frequency', 'Profit Margin %'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add traces
            fig_metrics.add_trace(
                go.Bar(x=segment_stats['segment'], y=segment_stats['avg_revenue'], 
                       name='Avg Revenue', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig_metrics.add_trace(
                go.Bar(x=segment_stats['segment'], y=segment_stats['avg_order_value'], 
                       name='Avg Order Value', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig_metrics.add_trace(
                go.Bar(x=segment_stats['segment'], y=segment_stats['avg_orders'], 
                       name='Avg Orders', marker_color='salmon'),
                row=2, col=1
            )
            
            fig_metrics.add_trace(
                go.Bar(x=segment_stats['segment'], y=segment_stats['avg_margin'], 
                       name='Avg Margin %', marker_color='gold'),
                row=2, col=2
            )
            
            fig_metrics.update_xaxes(tickangle=45)
            fig_metrics.update_layout(
                height=600, 
                showlegend=False,
                title_text="Customer Segment Performance Dashboard"
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # 4. Customer Journey Timeline
            st.subheader("Customer Purchase Patterns")
            
            # Monthly customer acquisition and retention
            monthly_customers = filtered_df.groupby([
                pd.Grouper(key='order_date', freq='M'),
                'CID'
            ]).size().reset_index().groupby('order_date')['CID'].nunique().reset_index()
            monthly_customers.columns = ['month', 'active_customers']
            
            # Calculate new vs returning customers
            customer_first_order = filtered_df.groupby('CID')['order_date'].min().reset_index()
            customer_first_order.columns = ['CID', 'first_order_date']
            
            df_with_first_order = filtered_df.merge(customer_first_order, on='CID')
            df_with_first_order['is_new_customer'] = (
                df_with_first_order['order_date'].dt.to_period('M') == 
                df_with_first_order['first_order_date'].dt.to_period('M')
            )
            
            df_with_first_order['customer_type'] = df_with_first_order['is_new_customer'].map({True: 'New', False: 'Returning'})
            
            monthly_new_returning = df_with_first_order.groupby([
                pd.Grouper(key='order_date', freq='M'),
                'customer_type'
            ])['CID'].nunique().reset_index()
            
            fig_timeline = px.bar(
                monthly_new_returning, 
                x='order_date', 
                y='CID',
                color='customer_type',
                title="New vs Returning Customers Over Time",
                labels={'CID': 'Number of Customers', 'customer_type': 'Customer Type'},
                color_discrete_map={'New': '#FF6B6B', 'Returning': '#4ECDC4'}
            )
            
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating enhanced customer analysis: {e}")
            # Fallback to original visualization
            try:
                fig = px.violin(filtered_df, y='customer_age', x='GEN', 
                                box=True, points="all",
                                title="Age Distribution by Gender")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e2:
                st.error(f"Error creating fallback chart: {e2}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for Customer Insights.")

with tab3:
    if not filtered_df.empty:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Only display one Profit Margin by Category chart
                margin_by_cat = filtered_df.groupby('CAT')['Margin'].mean().reset_index()
                fig = px.bar(margin_by_cat, x='CAT', y='Margin',
                             title="Profit Margin by Category",
                             color='Margin', color_continuous_scale='RdYlGn')
                fig.update_layout(
                    title={
                        'text': "Profit Margin by Category",
                        'x': 0.5,
                        'xanchor': 'center'
                    }
                )
                st.plotly_chart(fig, use_container_width=True, key="margin_chart")
            except Exception as e:
                st.error(f"Error creating margin chart: {e}")
        
        # Add dropdown filter for Sunburst chart
        sunburst_cats = sorted(filtered_df['CAT'].dropna().unique())
        selected_sunburst_cat = st.selectbox("Select Category for Sunburst", options=["All"] + sunburst_cats, index=0, key="sunburst_cat")
        if selected_sunburst_cat != "All":
            sunburst_df = filtered_df[filtered_df['CAT'] == selected_sunburst_cat]
        else:
            sunburst_df = filtered_df

        # Move Sunburst chart below the dropdown, outside of columns
        try:
            fig_sunburst = px.sunburst(sunburst_df, path=['CAT', 'SUBCAT', 'prd_nm'], 
                                      values='Revenue', title="Product Hierarchy")
            st.plotly_chart(fig_sunburst, use_container_width=True, key="sunburst_chart")
        except Exception as e:
            st.error(f"Error creating sunburst chart: {e}")

        # Dropdown to select visualization type for Product Characteristics
        chart_type = st.selectbox("Select Product Characteristics Visualization", ["Bubble Plot", "Parallel Coordinates"])
        try:
            numeric_cols = ['prd_cost', 'sls_price', 'Margin', 'sls_quantity']
            available_cols = [col for col in numeric_cols if col in filtered_df.columns]
            if len(available_cols) >= 2:
                if chart_type == "Bubble Plot":
                    # Bubble plot: x=sls_price, y=Margin, size=sls_quantity, color=prd_cost
                    fig_bubble = px.scatter(
                        filtered_df.dropna(subset=['sls_price', 'Margin', 'sls_quantity', 'prd_cost']),
                        x='sls_price', y='Margin',
                        size='sls_quantity', color='prd_cost',
                        hover_name='prd_nm',
                        title="Product Characteristics",
                        color_continuous_scale='Blues',
                        size_max=30
                    )
                    st.plotly_chart(fig_bubble, use_container_width=True, key="bubble_chart")
                else:
                    # Parallel Coordinates
                    fig_parallel = px.parallel_coordinates(
                        filtered_df.dropna(subset=available_cols),
                        dimensions=available_cols,
                        color='prd_cost',
                        title="Product Characteristics",
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig_parallel, use_container_width=True, key="parallel_chart")
            else:
                st.info("Insufficient numeric data for selected chart.")
        except Exception as e:
            st.error(f"Error creating product characteristics chart: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for Product Analytics.")

with tab4:
    if not filtered_df.empty:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fulfillment = filtered_df.groupby(['MAINTENANCE', 'CAT'])['order_to_ship_days'].mean().reset_index()
                fig = px.bar(fulfillment, x='CAT', y='order_to_ship_days',
                             color='MAINTENANCE', barmode='stack',
                             title="Average Fulfillment Time by Category")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating fulfillment chart: {e}")
        
        with col2:
            try:
                # Fixed daily orders resampling
                daily_orders = filtered_df.set_index('order_date').resample('D')['sls_ord_num'].nunique().reset_index()
                daily_orders['weekday'] = daily_orders['order_date'].dt.weekday
                daily_orders['month'] = daily_orders['order_date'].dt.month
                
                fig = px.density_heatmap(daily_orders, x='weekday', y='month',
                                         z='sls_ord_num', 
                                         title="Order Density Calendar")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating order density chart: {e}")
        
        try:
            metrics = filtered_df.groupby('CNTRY').agg({
                'order_to_ship_days': 'mean',
                'ship_delay': 'mean',
                'Margin': 'mean'
            }).reset_index()
            fig = px.line_polar(metrics, r='order_to_ship_days', theta='CNTRY',
                                line_close=True, title="Operational Metrics by Country")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating operational metrics chart: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for Operations.")

# Data explorer section
with st.expander("🔍 Data Explorer", expanded=False):
    if not filtered_df.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_field = st.selectbox("Sort by", options=filtered_df.columns)
        with col2:
            sort_order = st.radio("Order", ["⬆️ Asc", "⬇️ Desc"], horizontal=True)
        
        try:
            sorted_df = filtered_df.sort_values(
                sort_field, 
                ascending=(sort_order == "⬆️ Asc"),
                ignore_index=True
            )
            
            # Use display columns for the dataframe view
            display_df = sorted_df.copy()
            if 'order_date_display' in display_df.columns:
                display_df['order_date'] = display_df['order_date_display']
            if 'ship_date_display' in display_df.columns:
                display_df['ship_date'] = display_df['ship_date_display']
            if 'due_date_display' in display_df.columns:
                display_df['due_date'] = display_df['due_date_display']
            
            # Remove display columns
            display_df = display_df.drop(columns=[col for col in display_df.columns if col.endswith('_display')])
            
            st.dataframe(display_df.head(100), use_container_width=True)
            st.caption(f"Showing 100 of {len(sorted_df):,} records")
            
            # Export options
            export_format = st.radio("Export format", ["CSV", "JSON"], horizontal=True)
            if export_format == "CSV":
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download CSV", csv, "sales_data.csv")
            else:
                json = display_df.to_json(indent=2).encode('utf-8')
                st.download_button("💾 Download JSON", json, "sales_data.json")
        except Exception as e:
            st.error(f"Error in data explorer: {e}")
    else:
        st.info("No data available to explore.")

# Custom animations
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .plot-container {
            animation: fadeIn 0.5s ease;
        }
    </style>
""", unsafe_allow_html=True)

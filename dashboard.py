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
    page_icon="💲",
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
    <style>
        @keyframes fadeInSlide {
            0% {
                opacity: 0;
                transform: translateY(-20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .custom-title {
            text-align: center;
            padding: 1.5rem 0;
            position: sticky;
            top: 0;
            z-index: 1000;
            background: linear-gradient(180deg, #f8f9fa 90%, transparent);
            animation: fadeInSlide 1s ease-out;
        }

        .custom-title h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #6c5ce7, #a66efa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .custom-underline {
            height: 3px;
            background: linear-gradient(90deg, #6c5ce7 0%, #a66efa 100%);
            width: 80px;
            margin: 0.75rem auto 0;
            border-radius: 4px;
            animation: fadeInSlide 1.2s ease-out;
        }
    </style>

    <div class="custom-title">
        <h1>Wowowowow Sales Dashboard</h1>
        <div class="custom-underline"></div>
    </div>
""", unsafe_allow_html=True)


# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
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
        
        # Enhanced Customer Analysis Section (SIMPLIFIED)
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
            
            # 1. Key Customer Insights
            st.subheader("Key Customer Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                champions_count = len(customer_metrics[customer_metrics['segment'] == 'Champions'])
                champions_revenue = customer_metrics[customer_metrics['segment'] == 'Champions']['total_revenue'].sum()
                st.metric(
                    "🏆 Champions", 
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
            
            # 2. SIMPLIFIED VISUALIZATIONS
            st.subheader("Customer Segmentation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Customer Value by Segment (Bar Chart)
                segment_value = customer_metrics.groupby('segment').agg({
                    'total_revenue': 'mean',
                    'CID': 'count'
                }).reset_index().sort_values('total_revenue', ascending=False)
                
                fig_segment = px.bar(
                    segment_value, 
                    x='segment', 
                    y='total_revenue',
                    title="Average Customer Value by Segment",
                    labels={'total_revenue': 'Avg Revenue per Customer ($)', 'segment': 'Customer Segment'},
                    color='segment',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_segment.update_layout(showlegend=False)
                st.plotly_chart(fig_segment, use_container_width=True)
                
                # Add small multiples for segment distribution
                st.caption("Customer Distribution by Segment")
                segment_counts = customer_metrics['segment'].value_counts().reset_index()
                segment_counts.columns = ['segment', 'count']
                segment_counts['pct'] = (segment_counts['count'] / segment_counts['count'].sum()) * 100
                
                # Display as metric cards
                cols = st.columns(4)
                for i, row in segment_counts.head(4).iterrows():
                    with cols[i]:
                        st.metric(
                            label=row['segment'],
                            value=f"{row['count']}",
                            delta=f"{row['pct']:.1f}% of total"
                        )
            
            with col2:
                # Age vs Revenue (Binned Scatter Plot)
                # Create age bins
                customer_metrics['age_group'] = pd.cut(
                    customer_metrics['age'],
                    bins=[0, 20, 30, 40, 50, 60, 100],
                    labels=['<20', '20-29', '30-39', '40-49', '50-59', '60+']
                )
                
                age_revenue = customer_metrics.groupby(['age_group', 'gender']).agg({
                    'total_revenue': 'mean',
                    'CID': 'count'
                }).reset_index()
                
                fig_age = px.scatter(
                    age_revenue,
                    x='age_group',
                    y='total_revenue',
                    size='CID',
                    color='gender',
                    title="👥 Average Revenue by Age Group",
                    labels={
                        'total_revenue': 'Avg Revenue per Customer ($)',
                        'age_group': 'Age Group',
                        'CID': 'Number of Customers',
                        'gender': 'Gender'
                    },
                    size_max=30
                )
                fig_age.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_age, use_container_width=True)
                
                # Add gender distribution insight
                gender_dist = customer_metrics['gender'].value_counts(normalize=True).mul(100).round(1)
                st.caption(f"Customer Gender Distribution: {gender_dist.to_dict()}")
            
            # 3. Customer Segment Performance Table
            st.subheader("Segment Performance Summary")
            
            # Calculate segment statistics
            segment_stats = customer_metrics.groupby('segment').agg({
                'CID': 'count',
                'total_revenue': ['mean', 'sum'],
                'avg_order_value': 'mean',
                'total_orders': 'mean',
                'recency_days': 'mean',
                'profit_margin': 'mean'
            }).round(2).sort_values(('total_revenue', 'sum'), ascending=False)
            
            # Format the table
            segment_stats.columns = ['Customer Count', 'Avg Revenue', 'Total Revenue', 
                                   'Avg Order Value', 'Avg Orders', 'Avg Recency (days)', 'Avg Margin (%)']
            
            # Display as a styled table
            st.dataframe(
                segment_stats.style
                .background_gradient(subset=['Total Revenue'], cmap='Blues')
                .background_gradient(subset=['Avg Margin (%)'], cmap='RdYlGn')
                .format({
                    'Avg Revenue': "${:,.0f}",
                    'Total Revenue': "${:,.0f}",
                    'Avg Order Value': "${:,.0f}",
                    'Avg Margin (%)': "{:.1f}%"
                }),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error creating customer analysis: {e}")
            # Fallback to very simple visualizations
            try:
                st.subheader("Basic Customer Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    age_dist = px.histogram(
                        filtered_df, 
                        x='customer_age',
                        title="Customer Age Distribution",
                        nbins=20
                    )
                    st.plotly_chart(age_dist, use_container_width=True)
                
                with col2:
                    gender_dist = px.pie(
                        filtered_df, 
                        names='GEN',
                        title="Customer Gender Distribution"
                    )
                    st.plotly_chart(gender_dist, use_container_width=True)
            except Exception as e2:
                st.error(f"Error creating fallback charts: {e2}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No data available for Customer Insights.")

with tab3:
    if not filtered_df.empty:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                margin_by_cat = filtered_df.groupby('CAT')['Margin'].mean().reset_index()
                fig = px.bar(margin_by_cat, x='CAT', y='Margin',
                             title="Profit Margin by Category",
                             color='Margin', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating margin chart: {e}")
        
        with col2:
            try:
                # Simple stacked bar chart showing revenue by category and subcategory
                cat_subcat_rev = filtered_df.groupby(['CAT', 'SUBCAT'])['Revenue'].sum().reset_index()
                fig = px.bar(cat_subcat_rev, 
                             x='CAT', 
                             y='Revenue',
                             color='SUBCAT',
                             title="Revenue by Category & Subcategory",
                             labels={'Revenue': 'Total Revenue ($)'})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating product hierarchy chart: {e}")
        
        try:
            # Top products table - simple and effective
            st.subheader("Top Performing Products")
            top_products = filtered_df.groupby('prd_nm').agg({
                'Revenue': 'sum',
                'Margin': 'mean',
                'sls_quantity': 'sum'
            }).nlargest(10, 'Revenue').reset_index()
            
            # Format the numbers nicely
            top_products['Revenue'] = top_products['Revenue'].apply(lambda x: f"${x:,.0f}")
            top_products['Margin'] = top_products['Margin'].apply(lambda x: f"{x:.1f}%")
            top_products['sls_quantity'] = top_products['sls_quantity'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(
                top_products,
                column_config={
                    "prd_nm": "Product Name",
                    "Revenue": "Total Revenue",
                    "Margin": "Avg Margin",
                    "sls_quantity": "Units Sold"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Product performance scatter plot
            st.subheader("Product Performance Analysis")
            product_performance = filtered_df.groupby('prd_nm').agg({
                'Revenue': 'sum',
                'Margin': 'mean',
                'sls_quantity': 'sum'
            }).reset_index()
            
            fig = px.scatter(
                product_performance,
                x='sls_quantity',
                y='Revenue',
                size='Margin',
                color='Margin',
                hover_name='prd_nm',
                title="Product Performance: Quantity vs Revenue (Size = Margin)",
                labels={
                    'sls_quantity': 'Units Sold',
                    'Revenue': 'Total Revenue',
                    'Margin': 'Profit Margin (%)'
                },
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating product performance visualizations: {e}")
        
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
            # Simple bar charts for each metric by country
            metrics = filtered_df.groupby('CNTRY').agg({
                'order_to_ship_days': 'mean',
                'ship_delay': 'mean',
                'Margin': 'mean'
            }).reset_index()
            
            st.subheader("Operational Metrics by Country")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = px.bar(metrics, x='CNTRY', y='order_to_ship_days',
                             title="Avg Order to Ship Days",
                             color='order_to_ship_days',
                             color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(metrics, x='CNTRY', y='ship_delay',
                             title="Avg Ship Delay (Days)",
                             color='ship_delay',
                             color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.bar(metrics, x='CNTRY', y='Margin',
                             title="Avg Profit Margin (%)",
                             color='Margin',
                             color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
            
            # Add a simple table for the metrics
            st.subheader("Detailed Metrics by Country")
            metrics_display = metrics.copy()
            metrics_display['order_to_ship_days'] = metrics_display['order_to_ship_days'].round(1)
            metrics_display['ship_delay'] = metrics_display['ship_delay'].round(1)
            metrics_display['Margin'] = metrics_display['Margin'].round(1)
            
            st.dataframe(
                metrics_display.style
                .background_gradient(subset=['order_to_ship_days'], cmap='Blues')
                .background_gradient(subset=['ship_delay'], cmap='Reds')
                .background_gradient(subset=['Margin'], cmap='RdYlGn'),
                column_config={
                    "CNTRY": "Country",
                    "order_to_ship_days": "Avg Ship Time (Days)",
                    "ship_delay": "Avg Delay (Days)",
                    "Margin": "Avg Margin (%)"
                },
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error creating operational metrics charts: {e}")
        
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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import moi_utils as moi
import database as db

st.set_page_config(
    page_title="Media Opportunity Index (MOI)",
    page_icon="📊",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables."""
    if 'weight_presets' not in st.session_state:
        st.session_state.weight_presets = {}
    
    if 'current_weights' not in st.session_state:
        st.session_state.current_weights = {
            'w_rev_per_restaurant': 25,
            'w_pct_sales_search': 20,
            'w_meta_reach_opportunity': 10,
            'w_tiktok_reach_opportunity': 10,
            'w_untapped_digital_share': 15,
            'w_spend_opportunity': 20
        }
    
    if 'weight_toggles' not in st.session_state:
        st.session_state.weight_toggles = {
            'w_rev_per_restaurant': True,
            'w_pct_sales_search': True,
            'w_meta_reach_opportunity': True,
            'w_tiktok_reach_opportunity': True,
            'w_untapped_digital_share': True,
            'w_spend_opportunity': True
        }


def create_dma_template():
    """Create DMA template CSV with sample data."""
    template_data = pd.DataFrame({
        'DMA': ['New York', 'Los Angeles', 'Chicago', 'Dallas-Fort Worth', 'Houston'],
        'Revenue per Restaurant': ['$5,000', '$4,500', '$4,200', '$3,800', '$3,500'],
        '% Sales Search': ['25%', '30%', '22%', '28%', '24%'],
        '% Sales Google': ['40%', '45%', '38%', '42%', '39%'],
        'Ad Spend per Restaurant': ['$500', '$600', '$450', '$550', '$480'],
        'Meta Reach': ['0.65', '0.70', '0.60', '0.68', '0.62'],
        'TikTok Reach': ['0.55', '0.60', '0.50', '0.58', '0.52']
    })
    return template_data.to_csv(index=False).encode('utf-8')


def create_county_template():
    """Create County template CSV with sample data."""
    template_data = pd.DataFrame({
        'County': ['Los Angeles County', 'Cook County', 'Harris County', 'Maricopa County', 'San Diego County'],
        'Revenue per Restaurant': ['$4,500', '$4,200', '$3,500', '$3,800', '$4,000'],
        '% Sales Search': ['30%', '22%', '24%', '26%', '28%'],
        '% Sales Google': ['45%', '38%', '39%', '41%', '43%'],
        'Ad Spend per Restaurant': ['$600', '$450', '$480', '$520', '$550'],
        'DMA': ['Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'San Diego']
    })
    return template_data.to_csv(index=False).encode('utf-8')


def display_home():
    """Display home page with MOI concept description."""
    st.title("📊 Media Opportunity Index (MOI)")
    
    st.markdown("""
    ## What is the Media Opportunity Index?
    
    The **Media Opportunity Index (MOI)** is a composite metric that helps identify the most promising markets 
    for media investment by analyzing multiple performance and opportunity dimensions.
    
    ### Key Components
    
    The MOI combines six weighted components:
    
    1. **Revenue per Restaurant** (Direct) - Higher revenue indicates stronger market performance
    2. **% Sales from Search** (Direct) - Higher search sales show strong digital intent
    3. **Meta Reach Opportunity** (Inverted) - Lower Meta saturation means more room to grow reach
    4. **TikTok Reach Opportunity** (Inverted) - Lower TikTok saturation means more room to grow reach
    5. **Untapped Digital Share** (Inverted) - Lower Google share indicates opportunity to capture digital market
    6. **Spend Opportunity** (Inverted) - Lower current spend suggests efficiency potential
    
    ### How It Works
    
    1. **Upload** your DMA or County-level data (CSV format)
    2. **Map** your columns to the required fields
    3. **Customize** weights based on your strategic priorities
    4. **Compute** the MOI with percentile-based opportunity tiers
    5. **Download** results and insights
    
    ### Opportunity Tiers
    
    Markets are automatically categorized based on their MOI score:
    - **Exceptional** (≥99th percentile) - Top priority markets
    - **High** (66th-99th percentile) - Strong opportunity markets
    - **Moderate** (33rd-66th percentile) - Standard opportunity markets
    - **Lower** (<33rd percentile) - Lower priority markets
    
    ---
    
    ### 📥 CSV Templates
    
    Download these example CSV templates to see the required column format:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dma_template = create_dma_template()
        st.download_button(
            label="📄 Download DMA Template",
            data=dma_template,
            file_name="dma_template.csv",
            mime="text/csv",
            help="Example CSV with required columns for DMA analysis"
        )
        st.info("**DMA Template includes:** DMA, Revenue per Restaurant, % Sales Search, % Sales Google, Ad Spend per Restaurant, Meta Reach, TikTok Reach")
    
    with col2:
        county_template = create_county_template()
        st.download_button(
            label="📄 Download County Template",
            data=county_template,
            file_name="county_template.csv",
            mime="text/csv",
            help="Example CSV with required columns for County analysis (Meta/TikTok optional)"
        )
        st.info("**County Template includes:** County, Revenue per Restaurant, % Sales Search, % Sales Google, Ad Spend per Restaurant, DMA (optional)")
    
    st.markdown("""
    ---
    
    Choose a tab above to get started with **DMA** or **County** analysis.
    """)


def column_mapping_ui(df, analysis_type):
    """Create column mapping UI with auto-detection."""
    st.subheader("📋 Column Mapping")
    
    columns = [''] + list(df.columns)
    
    st.write("Map your CSV columns to the required fields:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grouping_key = analysis_type.lower()
        grouping_auto = moi.auto_detect_column(df.columns, f'grouping_{grouping_key}')
        grouping_default = columns.index(grouping_auto) if grouping_auto else 0
        grouping_col = st.selectbox(
            f"{analysis_type} (Grouping Key) *",
            columns,
            index=grouping_default,
            help=f"Select the column containing {analysis_type} names"
        )
        
        revenue_auto = moi.auto_detect_column(df.columns, 'revenue_per_restaurant')
        revenue_default = columns.index(revenue_auto) if revenue_auto else 0
        revenue_col = st.selectbox(
            "Revenue per Restaurant *",
            columns,
            index=revenue_default,
            help="Revenue or sales per restaurant/store"
        )
        
        search_auto = moi.auto_detect_column(df.columns, 'pct_sales_search')
        search_default = columns.index(search_auto) if search_auto else 0
        search_col = st.selectbox(
            "% Sales (Search Only) *",
            columns,
            index=search_default,
            help="Percentage of sales from search channels"
        )
    
    with col2:
        google_auto = moi.auto_detect_column(df.columns, 'pct_sales_google')
        google_default = columns.index(google_auto) if google_auto else 0
        google_col = st.selectbox(
            "% Sales (Google Total) *",
            columns,
            index=google_default,
            help="Percentage of sales from all Google channels"
        )
        
        spend_auto = moi.auto_detect_column(df.columns, 'ad_spend')
        spend_default = columns.index(spend_auto) if spend_auto else 0
        spend_col = st.selectbox(
            "Ad Spend per Restaurant *",
            columns,
            index=spend_default,
            help="Advertising spend per restaurant/store"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        meta_auto = moi.auto_detect_column(df.columns, 'meta_reach')
        meta_default = columns.index(meta_auto) if meta_auto else 0
        meta_label = "Meta Reach Saturation (optional for County)" if analysis_type == 'County' else "Meta Reach Saturation *"
        meta_col = st.selectbox(
            meta_label,
            columns,
            index=meta_default,
            help="Meta/Facebook reach saturation (0-1 or 0-100%)"
        )
    
    with col4:
        tiktok_auto = moi.auto_detect_column(df.columns, 'tiktok_reach')
        tiktok_default = columns.index(tiktok_auto) if tiktok_auto else 0
        tiktok_label = "TikTok Reach Saturation (optional for County)" if analysis_type == 'County' else "TikTok Reach Saturation *"
        tiktok_col = st.selectbox(
            tiktok_label,
            columns,
            index=tiktok_default,
            help="TikTok reach saturation (0-1 or 0-100%)"
        )
    
    if analysis_type == 'County':
        dma_auto = moi.auto_detect_column(df.columns, 'grouping_dma')
        dma_default = columns.index(dma_auto) if dma_auto else 0
        dma_col = st.selectbox(
            "DMA (for rollup - optional)",
            columns,
            index=dma_default,
            help="DMA column for County→DMA rollup analysis"
        )
    else:
        dma_col = None
    
    mapping = {
        'grouping': grouping_col,
        'revenue': revenue_col,
        'search': search_col,
        'google': google_col,
        'spend': spend_col,
        'meta': meta_col,
        'tiktok': tiktok_col,
        'dma': dma_col
    }
    
    if analysis_type == 'County':
        required_fields = ['grouping', 'revenue', 'search', 'google', 'spend']
    else:
        required_fields = ['grouping', 'revenue', 'search', 'google', 'spend', 'meta', 'tiktok']
    
    missing_fields = [field for field in required_fields if not mapping[field]]
    
    if missing_fields:
        st.warning(f"⚠️ Please map all required fields: {', '.join(missing_fields)}")
        return None
    
    return mapping


def weights_settings_ui():
    """Create weights and settings UI."""
    st.subheader("⚖️ Weights & Settings")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Component Weights** (adjust to prioritize different factors)")
        
        weight_cols = st.columns(6)
        
        with weight_cols[0]:
            toggle_rev = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_rev_per_restaurant'],
                key='toggle_rev'
            )
            weight_rev = st.slider(
                "Revenue/Restaurant",
                0, 100,
                st.session_state.current_weights['w_rev_per_restaurant'],
                disabled=not toggle_rev,
                key='slider_rev'
            )
            if not toggle_rev:
                weight_rev = 0
        
        with weight_cols[1]:
            toggle_search = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_pct_sales_search'],
                key='toggle_search'
            )
            weight_search = st.slider(
                "% Sales Search",
                0, 100,
                st.session_state.current_weights['w_pct_sales_search'],
                disabled=not toggle_search,
                key='slider_search'
            )
            if not toggle_search:
                weight_search = 0
        
        with weight_cols[2]:
            toggle_meta_reach = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_meta_reach_opportunity'],
                key='toggle_meta_reach'
            )
            weight_meta_reach = st.slider(
                "Meta Reach Opp",
                0, 100,
                st.session_state.current_weights['w_meta_reach_opportunity'],
                disabled=not toggle_meta_reach,
                key='slider_meta_reach'
            )
            if not toggle_meta_reach:
                weight_meta_reach = 0
        
        with weight_cols[3]:
            toggle_tiktok_reach = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_tiktok_reach_opportunity'],
                key='toggle_tiktok_reach'
            )
            weight_tiktok_reach = st.slider(
                "TikTok Reach Opp",
                0, 100,
                st.session_state.current_weights['w_tiktok_reach_opportunity'],
                disabled=not toggle_tiktok_reach,
                key='slider_tiktok_reach'
            )
            if not toggle_tiktok_reach:
                weight_tiktok_reach = 0
        
        with weight_cols[4]:
            toggle_digital = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_untapped_digital_share'],
                key='toggle_digital'
            )
            weight_digital = st.slider(
                "Untapped Digital",
                0, 100,
                st.session_state.current_weights['w_untapped_digital_share'],
                disabled=not toggle_digital,
                key='slider_digital'
            )
            if not toggle_digital:
                weight_digital = 0
        
        with weight_cols[5]:
            toggle_spend = st.checkbox(
                "Include",
                value=st.session_state.weight_toggles['w_spend_opportunity'],
                key='toggle_spend'
            )
            weight_spend = st.slider(
                "Spend Opportunity",
                0, 100,
                st.session_state.current_weights['w_spend_opportunity'],
                disabled=not toggle_spend,
                key='slider_spend'
            )
            if not toggle_spend:
                weight_spend = 0
        
        total_weight = weight_rev + weight_search + weight_meta_reach + weight_tiktok_reach + weight_digital + weight_spend
        
        if total_weight > 0:
            st.info(f"📊 Total Weight: {total_weight} → Normalized to 1.00")
        else:
            st.error("⚠️ At least one component must be enabled with weight > 0")
    
    with col2:
        st.write("**Tier Thresholds**")
        use_custom_thresholds = st.checkbox(
            "Override auto percentiles",
            value=False,
            help="Manually set tier thresholds instead of using calculated percentiles"
        )
        
        custom_thresholds = None
        if use_custom_thresholds:
            st.write("Set custom thresholds (0-1 scale):")
            p33_custom = st.number_input(
                "Lower → Moderate (33rd)",
                min_value=0.0, max_value=1.0, value=0.65, step=0.05
            )
            p66_custom = st.number_input(
                "Moderate → High (66th)",
                min_value=0.0, max_value=1.0, value=0.80, step=0.05
            )
            p99_custom = st.number_input(
                "High → Exceptional (99th)",
                min_value=0.0, max_value=1.0, value=0.95, step=0.05
            )
            custom_thresholds = {
                'p33': p33_custom,
                'p66': p66_custom,
                'p99': p99_custom
            }
    
    weights = {
        'w_rev_per_restaurant': weight_rev,
        'w_pct_sales_search': weight_search,
        'w_meta_reach_opportunity': weight_meta_reach,
        'w_tiktok_reach_opportunity': weight_tiktok_reach,
        'w_untapped_digital_share': weight_digital,
        'w_spend_opportunity': weight_spend
    }
    
    st.session_state.current_weights = weights
    st.session_state.weight_toggles = {
        'w_rev_per_restaurant': toggle_rev,
        'w_pct_sales_search': toggle_search,
        'w_meta_reach_opportunity': toggle_meta_reach,
        'w_tiktok_reach_opportunity': toggle_tiktok_reach,
        'w_untapped_digital_share': toggle_digital,
        'w_spend_opportunity': toggle_spend
    }
    
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        preset_name = st.text_input("Preset Name", placeholder="e.g., Growth Focus")
        if st.button("💾 Save Preset") and preset_name:
            st.session_state.weight_presets[preset_name] = weights.copy()
            st.success(f"Saved preset: {preset_name}")
    
    with preset_col2:
        if st.session_state.weight_presets:
            selected_preset = st.selectbox(
                "Load Preset",
                [''] + list(st.session_state.weight_presets.keys())
            )
            if selected_preset and st.button("📂 Load"):
                st.session_state.current_weights = st.session_state.weight_presets[selected_preset].copy()
                st.rerun()
    
    return weights, total_weight, custom_thresholds


def process_and_compute_moi(df, mapping, weights, custom_thresholds=None):
    """Process data and compute MOI."""
    warnings = []
    
    processed_df = pd.DataFrame()
    processed_df[mapping['grouping']] = df[mapping['grouping']]
    
    processed_df['revenue_per_restaurant'] = moi.clean_data_column(
        df[mapping['revenue']], 'currency'
    )
    
    processed_df['pct_sales_search'] = moi.clean_data_column(
        df[mapping['search']], 'percentage'
    )
    
    processed_df['pct_sales_google'] = moi.clean_data_column(
        df[mapping['google']], 'percentage'
    )
    
    processed_df['ad_spend_per_restaurant'] = moi.clean_data_column(
        df[mapping['spend']], 'currency'
    )
    
    if mapping.get('meta') and mapping['meta']:
        processed_df['meta_reach'] = moi.clean_data_column(
            df[mapping['meta']], 'reach'
        )
    else:
        processed_df['meta_reach'] = np.nan
    
    if mapping.get('tiktok') and mapping['tiktok']:
        processed_df['tiktok_reach'] = moi.clean_data_column(
            df[mapping['tiktok']], 'reach'
        )
    else:
        processed_df['tiktok_reach'] = np.nan
    
    if mapping.get('dma') and mapping['dma']:
        processed_df['dma'] = df[mapping['dma']]
    
    norm_components = {}
    max_values = {}
    
    if weights['w_rev_per_restaurant'] > 0:
        norm, max_val, warns = moi.normalize_direct(processed_df['revenue_per_restaurant'])
        norm_components['revenue_per_restaurant'] = norm
        max_values['revenue_per_restaurant'] = max_val
        warnings.extend([f"Revenue: {w}" for w in warns])
        processed_df['revenue_per_restaurant_norm'] = norm
    else:
        processed_df['revenue_per_restaurant_norm'] = 0
        norm_components['revenue_per_restaurant'] = pd.Series(0, index=processed_df.index)
    
    if weights['w_pct_sales_search'] > 0:
        norm, max_val, warns = moi.normalize_direct(processed_df['pct_sales_search'])
        norm_components['pct_sales_search'] = norm
        max_values['pct_sales_search'] = max_val
        warnings.extend([f"% Sales Search: {w}" for w in warns])
        processed_df['pct_sales_search_norm'] = norm
    else:
        processed_df['pct_sales_search_norm'] = 0
        norm_components['pct_sales_search'] = pd.Series(0, index=processed_df.index)
    
    if weights['w_meta_reach_opportunity'] > 0:
        norm, max_val, warns = moi.normalize_inverted(processed_df['meta_reach'])
        norm_components['meta_reach_opportunity'] = norm
        max_values['meta_reach'] = max_val
        warnings.extend([f"Meta Reach Opportunity: {w}" for w in warns])
        processed_df['meta_reach_opportunity_norm'] = norm
    else:
        processed_df['meta_reach_opportunity_norm'] = 0
        norm_components['meta_reach_opportunity'] = pd.Series(0, index=processed_df.index)
    
    if weights['w_tiktok_reach_opportunity'] > 0:
        norm, max_val, warns = moi.normalize_inverted(processed_df['tiktok_reach'])
        norm_components['tiktok_reach_opportunity'] = norm
        max_values['tiktok_reach'] = max_val
        warnings.extend([f"TikTok Reach Opportunity: {w}" for w in warns])
        processed_df['tiktok_reach_opportunity_norm'] = norm
    else:
        processed_df['tiktok_reach_opportunity_norm'] = 0
        norm_components['tiktok_reach_opportunity'] = pd.Series(0, index=processed_df.index)
    
    if weights['w_untapped_digital_share'] > 0:
        norm, max_val, warns = moi.normalize_inverted(processed_df['pct_sales_google'])
        norm_components['untapped_digital_share'] = norm
        max_values['pct_sales_google'] = max_val
        warnings.extend([f"Untapped Digital: {w}" for w in warns])
        processed_df['untapped_digital_norm'] = norm
    else:
        processed_df['untapped_digital_norm'] = 0
        norm_components['untapped_digital_share'] = pd.Series(0, index=processed_df.index)
    
    if weights['w_spend_opportunity'] > 0:
        norm, max_val, warns = moi.normalize_inverted(processed_df['ad_spend_per_restaurant'])
        norm_components['spend_opportunity'] = norm
        max_values['ad_spend_per_restaurant'] = max_val
        warnings.extend([f"Spend Opportunity: {w}" for w in warns])
        processed_df['spend_opportunity_norm'] = norm
    else:
        processed_df['spend_opportunity_norm'] = 0
        norm_components['spend_opportunity'] = pd.Series(0, index=processed_df.index)
    
    weight_mapping = {
        'revenue_per_restaurant': weights['w_rev_per_restaurant'],
        'pct_sales_search': weights['w_pct_sales_search'],
        'meta_reach_opportunity': weights['w_meta_reach_opportunity'],
        'tiktok_reach_opportunity': weights['w_tiktok_reach_opportunity'],
        'untapped_digital_share': weights['w_untapped_digital_share'],
        'spend_opportunity': weights['w_spend_opportunity']
    }
    
    processed_df['MOI'] = moi.compute_moi(norm_components, weight_mapping)
    
    processed_df['MOI_Index'] = moi.compute_moi_index(processed_df['MOI'])
    
    tiers, thresholds = moi.assign_opportunity_tiers(processed_df['MOI'], custom_thresholds)
    processed_df['Tier'] = tiers
    
    return processed_df, warnings, max_values, thresholds


def display_results(results_df, grouping_col, max_values, thresholds):
    """Display results with interactive table and charts."""
    st.subheader("📊 Results")
    
    if len(results_df) == 0:
        st.warning("No results to display")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Markets", len(results_df))
    with col2:
        exceptional = len(results_df[results_df['Tier'] == 'Exceptional'])
        st.metric("Exceptional", exceptional)
    with col3:
        high = len(results_df[results_df['Tier'] == 'High'])
        st.metric("High Opportunity", high)
    with col4:
        avg_moi = results_df['MOI'].mean()
        st.metric("Avg MOI", f"{avg_moi:.3f}")
    
    with st.expander("📈 Normalization Reference (Max Values)", expanded=False):
        st.write("**Max values used for normalization:**")
        max_df = pd.DataFrame([max_values]).T
        max_df.columns = ['Max Value']
        st.dataframe(max_df)
        
        st.write("**Percentile Thresholds:**")
        thresh_df = pd.DataFrame([thresholds]).T
        thresh_df.columns = ['Threshold']
        st.dataframe(thresh_df)
    
    st.write("**Interactive Results Table**")
    st.write("Sort by clicking column headers. Use filters in the sidebar.")
    
    display_cols = [grouping_col, 'MOI', 'MOI_Index', 'Tier',
                    'revenue_per_restaurant_norm', 'pct_sales_search_norm',
                    'reach_opportunity_norm', 'untapped_digital_norm', 'spend_opportunity_norm']
    
    display_cols = [col for col in display_cols if col in results_df.columns]
    
    display_df = results_df[display_cols].sort_values('MOI', ascending=False)
    
    st.dataframe(
        display_df.style.format({
            'MOI': '{:.4f}',
            'MOI_Index': '{:.2f}',
            'revenue_per_restaurant_norm': '{:.4f}',
            'pct_sales_search_norm': '{:.4f}',
            'reach_opportunity_norm': '{:.4f}',
            'untapped_digital_norm': '{:.4f}',
            'spend_opportunity_norm': '{:.4f}'
        }),
        use_container_width=True,
        height=400
    )
    
    st.subheader("📉 Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("**Top 15 Markets by MOI**")
        top_15 = results_df.nlargest(15, 'MOI')
        fig_bar = px.bar(
            top_15,
            x=grouping_col,
            y='MOI',
            color='Tier',
            color_discrete_map={
                'Exceptional': '#1f77b4',
                'High': '#2ca02c',
                'Moderate': '#ff7f0e',
                'Lower': '#d62728'
            },
            title="Top 15 Markets by MOI"
        )
        fig_bar.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with chart_col2:
        st.write("**Revenue vs MOI by Tier**")
        fig_scatter = px.scatter(
            results_df,
            x='revenue_per_restaurant',
            y='MOI',
            color='Tier',
            hover_data=[grouping_col],
            color_discrete_map={
                'Exceptional': '#1f77b4',
                'High': '#2ca02c',
                'Moderate': '#ff7f0e',
                'Lower': '#d62728'
            },
            title="Revenue per Restaurant vs MOI"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    return display_df


def download_buttons(results_df, analysis_type, mapping=None, weights=None, max_values=None, thresholds=None, custom_thresholds=None):
    """Create download buttons for CSV and XLSX."""
    st.subheader("⬇️ Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = moi.create_download_csv(results_df)
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"moi_results_{analysis_type.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        xlsx_data = moi.create_download_excel(results_df, f"MOI {analysis_type}")
        st.download_button(
            label="📊 Download Excel",
            data=xlsx_data,
            file_name=f"moi_results_{analysis_type.lower()}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        if weights and max_values and thresholds:
            pdf_data = moi.create_methodology_pdf(
                analysis_type=analysis_type,
                weights=weights,
                max_values=max_values,
                thresholds=thresholds,
                num_markets=len(results_df),
                custom_thresholds=custom_thresholds
            )
            st.download_button(
                label="📋 Methodology PDF",
                data=pdf_data,
                file_name=f"moi_methodology_{analysis_type.lower()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    if db.engine and mapping and weights:
        st.divider()
        st.subheader("💾 Save Historical Snapshot")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            snapshot_name = st.text_input(
                "Snapshot Name",
                value=f"{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                help="Name for this historical snapshot"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("💾 Save", type="secondary", use_container_width=True):
                if snapshot_name:
                    try:
                        db.save_moi_snapshot(
                            snapshot_name=snapshot_name,
                            analysis_type=analysis_type,
                            results_df=results_df,
                            grouping_col=mapping['grouping'],
                            weights=weights
                        )
                        st.success(f"✅ Snapshot '{snapshot_name}' saved successfully!")
                    except Exception as e:
                        st.error(f"❌ Error saving snapshot: {str(e)}")
                else:
                    st.warning("Please enter a snapshot name")


def county_dma_rollup(results_df, mapping, weights, custom_thresholds=None):
    """Perform County to DMA rollup."""
    st.subheader("🔄 County → DMA Rollup")
    
    if not mapping.get('dma') or not mapping['dma'] or 'dma' not in results_df.columns:
        st.info("ℹ️ No DMA column mapped. Rollup not available.")
        return None
    
    st.write("Aggregate county-level data to DMA level:")
    
    col1, col2 = st.columns(2)
    with col1:
        revenue_agg = st.radio(
            "Revenue Aggregation",
            ['median', 'mean'],
            horizontal=True
        )
    with col2:
        spend_agg = st.radio(
            "Spend Aggregation",
            ['median', 'mean'],
            horizontal=True
        )
    
    if st.button("🔄 Compute DMA Rollup"):
        with st.spinner("Aggregating to DMA level..."):
            dma_df = moi.aggregate_county_to_dma(
                results_df,
                county_col=mapping['grouping'],
                dma_col='dma',
                revenue_col='revenue_per_restaurant',
                pct_search_col='pct_sales_search',
                pct_google_col='pct_sales_google',
                ad_spend_col='ad_spend_per_restaurant',
                meta_reach_col='meta_reach' if 'meta_reach' in results_df.columns else None,
                tiktok_reach_col='tiktok_reach' if 'tiktok_reach' in results_df.columns else None,
                revenue_agg=revenue_agg,
                spend_agg=spend_agg
            )
            
            temp_mapping = mapping.copy()
            temp_mapping['grouping'] = 'dma'
            
            dma_results, dma_warnings, dma_max_values, dma_thresholds = process_and_compute_moi(
                dma_df,
                temp_mapping,
                weights,
                custom_thresholds
            )
            
            if dma_warnings:
                st.warning("⚠️ Warnings during DMA rollup:\n" + "\n".join(dma_warnings))
            
            display_results(dma_results, 'dma', dma_max_values, dma_thresholds)
            
            st.write("**Download DMA Rollup Results**")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = moi.create_download_csv(dma_results)
                st.download_button(
                    label="📄 Download DMA CSV",
                    data=csv_data,
                    file_name=f"moi_dma_rollup_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                xlsx_data = moi.create_download_excel(dma_results, "MOI DMA Rollup")
                st.download_button(
                    label="📊 Download DMA Excel",
                    data=xlsx_data,
                    file_name=f"moi_dma_rollup_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            return dma_results
    
    return None


def analysis_tab(analysis_type):
    """Create analysis tab for DMA or County."""
    st.header(f"{analysis_type} Analysis")
    
    batch_mode = st.checkbox(
        "📦 Batch Processing Mode",
        value=False,
        help="Upload multiple CSV files for combined analysis",
        key=f'batch_mode_{analysis_type}'
    )
    
    if batch_mode:
        uploaded_files = st.file_uploader(
            f"Upload {analysis_type} CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            key=f'upload_{analysis_type}_batch'
        )
    else:
        uploaded_file = st.file_uploader(
            f"Upload {analysis_type} CSV",
            type=['csv'],
            key=f'upload_{analysis_type}'
        )
    
    if batch_mode:
        if uploaded_files and len(uploaded_files) > 0:
            try:
                st.success(f"✅ Loaded {len(uploaded_files)} CSV file(s)")
                
                combined_df = pd.concat([pd.read_csv(f) for f in uploaded_files], ignore_index=True)
                
                st.info(f"📊 Combined dataset: {len(combined_df)} total rows from {len(uploaded_files)} files")
                
                with st.expander("📋 Combined Data Preview (first 5 rows)", expanded=True):
                    st.dataframe(combined_df.head(), use_container_width=True)
                
                st.divider()
                
                mapping = column_mapping_ui(combined_df, analysis_type)
                
                if mapping:
                    st.divider()
                    
                    weights, total_weight, custom_thresholds = weights_settings_ui()
                    
                    st.divider()
                    
                    if total_weight > 0:
                        if st.button(f"🚀 Compute MOI for Combined {analysis_type}", type="primary", use_container_width=True):
                            with st.spinner("Computing MOI for combined dataset..."):
                                results_df, warnings, max_values, thresholds = process_and_compute_moi(
                                    combined_df, mapping, weights, custom_thresholds
                                )
                                
                                if warnings:
                                    with st.expander("⚠️ Data Warnings", expanded=False):
                                        for warning in warnings:
                                            st.warning(warning)
                                
                                st.divider()
                                
                                display_results(results_df, mapping['grouping'], max_values, thresholds)
                                
                                st.divider()
                                
                                download_buttons(results_df, f"{analysis_type}_batch", mapping, weights, max_values, thresholds, custom_thresholds)
                                
                                if analysis_type == 'County' and mapping.get('dma'):
                                    st.divider()
                                    county_dma_rollup(results_df, mapping, weights, custom_thresholds)
                    else:
                        st.error("⚠️ Please enable at least one component with weight > 0")
            
            except Exception as e:
                st.error(f"❌ Error processing batch files: {str(e)}")
                st.write("Please check your CSV formats and ensure they have compatible columns.")
    
    elif uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
            
            with st.expander("📋 Data Preview (first 5 rows)", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            st.divider()
            
            mapping = column_mapping_ui(df, analysis_type)
            
            if mapping:
                st.divider()
                
                weights, total_weight, custom_thresholds = weights_settings_ui()
                
                st.divider()
                
                if total_weight > 0:
                    if st.button(f"🚀 Compute MOI for {analysis_type}", type="primary", use_container_width=True):
                        with st.spinner("Computing MOI..."):
                            results_df, warnings, max_values, thresholds = process_and_compute_moi(
                                df, mapping, weights, custom_thresholds
                            )
                            
                            if warnings:
                                with st.expander("⚠️ Data Warnings", expanded=False):
                                    for warning in warnings:
                                        st.warning(warning)
                            
                            st.divider()
                            
                            display_results(results_df, mapping['grouping'], max_values, thresholds)
                            
                            st.divider()
                            
                            download_buttons(results_df, analysis_type, mapping, weights, max_values, thresholds, custom_thresholds)
                            
                            if analysis_type == 'County' and mapping.get('dma'):
                                st.divider()
                                county_dma_rollup(results_df, mapping, weights, custom_thresholds)
                else:
                    st.error("⚠️ Please enable at least one component with weight > 0")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please check your CSV format and try again.")


def historical_comparison_tab():
    """Historical comparison tab for tracking MOI changes over time."""
    st.header("📈 Historical Comparison")
    
    if not db.engine:
        st.warning("⚠️ Database not configured. Historical comparison requires database connectivity.")
        return
    
    st.write("Compare MOI snapshots over time to track market opportunity changes.")
    
    try:
        snapshots = db.get_snapshot_names()
        
        if not snapshots:
            st.info("ℹ️ No snapshots saved yet. Run an analysis and save a snapshot to use this feature.")
            return
        
        snapshot_options = [f"{s['name']} ({s['type']}) - {s['date'].strftime('%Y-%m-%d %H:%M')}" for s in snapshots]
        snapshot_map = {f"{s['name']} ({s['type']}) - {s['date'].strftime('%Y-%m-%d %H:%M')}": s['name'] for s in snapshots}
        
        col1, col2 = st.columns(2)
        
        with col1:
            baseline_snapshot = st.selectbox(
                "Baseline Snapshot (Earlier)",
                snapshot_options,
                help="Select the earlier snapshot for comparison"
            )
        
        with col2:
            current_snapshot = st.selectbox(
                "Current Snapshot (Later)",
                snapshot_options,
                index=min(1, len(snapshot_options) - 1),
                help="Select the later snapshot for comparison"
            )
        
        if st.button("🔍 Compare Snapshots", type="primary"):
            with st.spinner("Comparing snapshots..."):
                baseline_name = snapshot_map[baseline_snapshot]
                current_name = snapshot_map[current_snapshot]
                
                comparison_df = db.compare_snapshots(baseline_name, current_name)
                
                if comparison_df is None or len(comparison_df) == 0:
                    st.error("❌ Could not compare snapshots. Please check the data.")
                    return
                
                st.subheader("📊 Comparison Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    improved = len(comparison_df[comparison_df['MOI_Change'] > 0])
                    st.metric("Markets Improved", improved)
                with col2:
                    declined = len(comparison_df[comparison_df['MOI_Change'] < 0])
                    st.metric("Markets Declined", declined)
                with col3:
                    avg_change = comparison_df['MOI_Change'].mean()
                    st.metric("Avg MOI Change", f"{avg_change:+.4f}")
                with col4:
                    max_gain = comparison_df['MOI_Change'].max()
                    st.metric("Max Gain", f"{max_gain:+.4f}")
                
                st.write("**Detailed Comparison Table**")
                display_df = comparison_df.sort_values('MOI_Change', ascending=False)
                st.dataframe(
                    display_df.style.format({
                        'MOI_baseline': '{:.4f}',
                        'MOI_current': '{:.4f}',
                        'MOI_Change': '{:+.4f}',
                        'MOI_Index_baseline': '{:.2f}',
                        'MOI_Index_current': '{:.2f}',
                        'MOI_Index_Change': '{:+.2f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                st.subheader("📉 Change Visualization")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write("**Top 15 Biggest Changes**")
                    top_changes = display_df.head(15)
                    fig_changes = px.bar(
                        top_changes,
                        x='grouping_key',
                        y='MOI_Change',
                        color='MOI_Change',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title="MOI Change by Market"
                    )
                    fig_changes.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_changes, use_container_width=True)
                
                with chart_col2:
                    st.write("**Change Distribution**")
                    fig_dist = px.histogram(
                        comparison_df,
                        x='MOI_Change',
                        nbins=30,
                        title="Distribution of MOI Changes"
                    )
                    fig_dist.update_layout(height=400)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.divider()
                
                csv_data = moi.create_download_csv(comparison_df)
                st.download_button(
                    label="📄 Download Comparison CSV",
                    data=csv_data,
                    file_name=f"moi_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        st.divider()
        st.subheader("🗑️ Manage Snapshots")
        
        delete_snapshot = st.selectbox(
            "Select snapshot to delete",
            snapshot_options,
            key="delete_snapshot"
        )
        
        if st.button("🗑️ Delete Snapshot", type="secondary"):
            snapshot_to_delete = snapshot_map[delete_snapshot]
            try:
                db.delete_snapshot(snapshot_to_delete)
                st.success(f"✅ Deleted snapshot: {snapshot_to_delete}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error deleting snapshot: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Error loading snapshots: {str(e)}")


def main():
    """Main application."""
    init_session_state()
    
    tab_home, tab_dma, tab_county, tab_history = st.tabs(["🏠 Home", "📍 DMA Analysis", "🗺️ County Analysis", "📈 Historical"])
    
    with tab_home:
        display_home()
    
    with tab_dma:
        analysis_tab('DMA')
    
    with tab_county:
        analysis_tab('County')
    
    with tab_history:
        historical_comparison_tab()


if __name__ == "__main__":
    main()

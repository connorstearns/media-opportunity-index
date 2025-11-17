import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import io

COLUMN_VARIANTS = {
    'grouping_dma': ['DMA', 'DMA Region', 'dma', 'dma region'],
    'grouping_county': ['County', 'County (Matched)', 'county', 'county matched'],
    'revenue_per_restaurant': [
        'Revenue per Restaurant', 'Revenue per Store', 'Sales per Resto',
        'revenue per restaurant', 'revenue per store', 'sales per resto',
        'Rev per Restaurant', 'Rev per Store'
    ],
    'pct_sales_search': [
        '% Sales (Search Only)', '% Sales (Search)', '% Search Rev',
        'pct sales search only', 'pct sales search', 'pct search rev',
        'Sales Search %', 'Search Sales %'
    ],
    'pct_sales_google': [
        '% Sales (Google Total)', '% Sales (Google All)', '% Google Total',
        'pct sales google total', 'pct sales google all', 'pct google total',
        'Sales Google %', 'Google Sales %'
    ],
    'ad_spend': [
        'Ad Spend per Restaurant', 'Spend per Resto', 'Ad Spend per Store',
        'ad spend per restaurant', 'spend per resto', 'ad spend per store',
        'Spend per Restaurant'
    ],
    'meta_reach': [
        'Meta Reach Saturation', 'Max Meta Reach Saturation', 'Meta Reach',
        'meta reach saturation', 'max meta reach saturation', 'meta reach',
        'Meta Saturation'
    ],
    'tiktok_reach': [
        'TikTok Reach Saturation', 'Max TikTok Reach Saturation', 'TikTok Reach',
        'tiktok reach saturation', 'max tiktok reach saturation', 'tiktok reach',
        'TikTok Saturation'
    ]
}


def auto_detect_column(columns: List[str], field_type: str) -> Optional[str]:
    """
    Auto-detect column mapping based on predefined variants.
    
    Args:
        columns: List of column names from the CSV
        field_type: Type of field to detect (e.g., 'revenue_per_restaurant')
    
    Returns:
        Detected column name or None
    """
    if field_type not in COLUMN_VARIANTS:
        return None
    
    variants = COLUMN_VARIANTS[field_type]
    for col in columns:
        if col in variants:
            return col
    return None


def parse_currency(value) -> float:
    """
    Parse currency values by removing $ and commas.
    
    Args:
        value: Input value (str, float, or int)
    
    Returns:
        Parsed float value
    """
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        cleaned = value.replace('$', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    
    return np.nan


def parse_percentage(value) -> float:
    """
    Parse percentage values by removing % sign.
    Returns as decimal (e.g., 25% -> 0.25).
    
    Args:
        value: Input value (str, float, or int)
    
    Returns:
        Parsed float value as decimal
    """
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        val = float(value)
        if val > 1:
            return val / 100.0
        return val
    
    if isinstance(value, str):
        if '%' in value:
            cleaned = value.replace('%', '').strip()
            try:
                return float(cleaned) / 100.0
            except ValueError:
                return np.nan
        else:
            try:
                val = float(value)
                if val > 1:
                    return val / 100.0
                return val
            except ValueError:
                return np.nan
    
    return np.nan


def normalize_reach_rate(value) -> float:
    """
    Normalize reach saturation to 0-1 range.
    If values appear to be in 0-100 range, divide by 100.
    
    Args:
        value: Input value
    
    Returns:
        Normalized value between 0 and 1
    """
    if pd.isna(value):
        return np.nan
    
    try:
        val = float(value)
        if val > 1:
            return val / 100.0
        return val
    except (ValueError, TypeError):
        return np.nan


def clean_data_column(series: pd.Series, data_type: str) -> pd.Series:
    """
    Clean a data column based on its type.
    
    Args:
        series: Input pandas Series
        data_type: Type of data ('currency', 'percentage', 'reach', 'numeric')
    
    Returns:
        Cleaned pandas Series
    """
    if data_type == 'currency':
        return series.apply(parse_currency)
    elif data_type == 'percentage':
        return series.apply(parse_percentage)
    elif data_type == 'reach':
        return series.apply(normalize_reach_rate)
    elif data_type == 'numeric':
        return pd.to_numeric(series, errors='coerce')
    else:
        return series


def compute_reach_blend(
    meta_reach: pd.Series,
    tiktok_reach: pd.Series,
    method: str = 'average',
    meta_weight: float = 0.5
) -> pd.Series:
    """
    Compute reach blend from Meta and TikTok saturation.
    
    Args:
        meta_reach: Meta reach saturation series (0-1)
        tiktok_reach: TikTok reach saturation series (0-1)
        method: 'average', 'weighted', or 'max'
        meta_weight: Weight for Meta in weighted average (0-1)
    
    Returns:
        Blended reach series
    """
    if method == 'average':
        return (meta_reach + tiktok_reach) / 2
    elif method == 'weighted':
        tiktok_weight = 1 - meta_weight
        return meta_reach * meta_weight + tiktok_reach * tiktok_weight
    elif method == 'max':
        return pd.concat([meta_reach, tiktok_reach], axis=1).max(axis=1)
    else:
        return (meta_reach + tiktok_reach) / 2


def normalize_direct(series: pd.Series) -> Tuple[pd.Series, float, List[str]]:
    """
    Normalize a series where higher values are better.
    Formula: value / max(value)
    
    Args:
        series: Input series
    
    Returns:
        Tuple of (normalized series, max value, warnings list)
    """
    warnings = []
    
    valid_values = series.dropna()
    if len(valid_values) == 0:
        warnings.append("All values are missing")
        return pd.Series(0, index=series.index), 0, warnings
    
    max_val = valid_values.max()
    
    if max_val == 0 or pd.isna(max_val):
        warnings.append("All values are zero or constant")
        return pd.Series(0, index=series.index), 0, warnings
    
    normalized = series / max_val
    normalized = normalized.fillna(0)
    
    return normalized, max_val, warnings


def normalize_inverted(series: pd.Series) -> Tuple[pd.Series, float, List[str]]:
    """
    Normalize a series where lower values represent more opportunity.
    Formula: 1 - (value / max(value))
    
    Args:
        series: Input series
    
    Returns:
        Tuple of (normalized series, max value, warnings list)
    """
    warnings = []
    
    valid_values = series.dropna()
    if len(valid_values) == 0:
        warnings.append("All values are missing")
        return pd.Series(0, index=series.index), 0, warnings
    
    max_val = valid_values.max()
    
    if max_val == 0 or pd.isna(max_val):
        warnings.append("All values are zero or constant")
        return pd.Series(0, index=series.index), 0, warnings
    
    normalized = 1 - (series / max_val)
    normalized = normalized.fillna(0)
    
    return normalized, max_val, warnings


def compute_moi(
    normalized_components: Dict[str, pd.Series],
    weights: Dict[str, float]
) -> pd.Series:
    """
    Compute Media Opportunity Index from normalized components and weights.
    
    Args:
        normalized_components: Dictionary of normalized component series
        weights: Dictionary of weights (should sum to 1)
    
    Returns:
        MOI series
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.Series(0, index=normalized_components[list(normalized_components.keys())[0]].index)
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    moi = pd.Series(0.0, index=normalized_components[list(normalized_components.keys())[0]].index)
    
    for component, weight in normalized_weights.items():
        if component in normalized_components:
            moi += normalized_components[component] * weight
    
    return moi


def assign_opportunity_tiers(moi: pd.Series, custom_thresholds: Optional[Dict[str, float]] = None) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Assign opportunity tiers based on percentiles or custom thresholds.
    
    Tiers:
    - Lower: < 33rd percentile
    - Moderate: 33rd-66th percentile
    - High: 66th-99th percentile
    - Exceptional: >= 99th percentile
    
    Args:
        moi: MOI series
        custom_thresholds: Optional dict with custom threshold values {'p33': val, 'p66': val, 'p99': val}
    
    Returns:
        Tuple of (tier series, percentile thresholds dict)
    """
    valid_moi = moi.dropna()
    
    if custom_thresholds:
        thresholds = custom_thresholds
    elif len(valid_moi) <= 2:
        thresholds = {
            'p33': 0.65,
            'p66': 0.80,
            'p99': 0.95
        }
    else:
        thresholds = {
            'p33': valid_moi.quantile(0.33),
            'p66': valid_moi.quantile(0.66),
            'p99': valid_moi.quantile(0.99)
        }
    
    def assign_tier(value):
        if pd.isna(value):
            return 'N/A'
        elif value >= thresholds['p99']:
            return 'Exceptional'
        elif value >= thresholds['p66']:
            return 'High'
        elif value >= thresholds['p33']:
            return 'Moderate'
        else:
            return 'Lower'
    
    tiers = moi.apply(assign_tier)
    
    return tiers, thresholds


def compute_moi_index(moi: pd.Series) -> pd.Series:
    """
    Compute MOI Index (0-100 scale).
    Formula: 100 * (MOI - min) / (max - min)
    
    Args:
        moi: MOI series (0-1)
    
    Returns:
        MOI Index series (0-100)
    """
    valid_moi = moi.dropna()
    
    if len(valid_moi) == 0:
        return pd.Series(0, index=moi.index)
    
    min_moi = valid_moi.min()
    max_moi = valid_moi.max()
    
    if max_moi == min_moi:
        return pd.Series(50, index=moi.index)
    
    moi_index = 100 * (moi - min_moi) / (max_moi - min_moi)
    
    return moi_index


def aggregate_county_to_dma(
    df: pd.DataFrame,
    county_col: str,
    dma_col: str,
    revenue_col: str,
    pct_search_col: str,
    pct_google_col: str,
    ad_spend_col: str,
    meta_reach_col: Optional[str],
    tiktok_reach_col: Optional[str],
    revenue_agg: str = 'median',
    spend_agg: str = 'median'
) -> pd.DataFrame:
    """
    Aggregate county-level data to DMA level.
    
    Args:
        df: DataFrame with county-level data
        county_col: County column name
        dma_col: DMA column name
        revenue_col: Revenue per restaurant column
        pct_search_col: % Sales Search column
        pct_google_col: % Sales Google column
        ad_spend_col: Ad spend column
        meta_reach_col: Meta reach column (optional)
        tiktok_reach_col: TikTok reach column (optional)
        revenue_agg: Aggregation method for revenue ('median' or 'mean')
        spend_agg: Aggregation method for spend ('median' or 'mean')
    
    Returns:
        Aggregated DataFrame at DMA level
    """
    agg_dict = {
        revenue_col: revenue_agg,
        pct_search_col: 'mean',
        pct_google_col: 'mean',
        ad_spend_col: spend_agg
    }
    
    if meta_reach_col and meta_reach_col in df.columns:
        agg_dict[meta_reach_col] = 'mean'
    
    if tiktok_reach_col and tiktok_reach_col in df.columns:
        agg_dict[tiktok_reach_col] = 'mean'
    
    dma_df = df.groupby(dma_col).agg(agg_dict).reset_index()
    
    return dma_df


def create_download_csv(df: pd.DataFrame) -> bytes:
    """
    Create CSV file for download.
    
    Args:
        df: DataFrame to export
    
    Returns:
        CSV data as bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def create_download_excel(df: pd.DataFrame, sheet_name: str = 'MOI Results') -> bytes:
    """
    Create Excel file for download.
    
    Args:
        df: DataFrame to export
        sheet_name: Name of the Excel sheet
    
    Returns:
        Excel data as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


def create_methodology_pdf(
    analysis_type: str,
    weights: Dict[str, float],
    reach_method: str,
    max_values: Dict[str, float],
    thresholds: Dict[str, float],
    num_markets: int,
    custom_thresholds: Optional[Dict[str, float]] = None
) -> bytes:
    """
    Create a detailed PDF methodology report.
    
    Args:
        analysis_type: Type of analysis (DMA or County)
        weights: Weight configuration
        reach_method: Reach blend method used
        max_values: Max values used for normalization
        thresholds: Percentile thresholds
        num_markets: Number of markets analyzed
        custom_thresholds: Custom thresholds if used
    
    Returns:
        PDF data as bytes
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from datetime import datetime
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=10
    )
    
    story = []
    
    story.append(Paragraph("Media Opportunity Index (MOI)", title_style))
    story.append(Paragraph("Methodology Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis Type:</b> {analysis_type}", styles['Normal']))
    story.append(Paragraph(f"<b>Markets Analyzed:</b> {num_markets}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("MOI Formula", heading_style))
    story.append(Paragraph(
        "The Media Opportunity Index is calculated as a weighted composite score combining five normalized components:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    
    formula_text = """
    <b>MOI = w₁ × Revenue_norm + w₂ × Search_Sales_norm + w₃ × Reach_Opportunity_norm 
          + w₄ × Untapped_Digital_norm + w₅ × Spend_Opportunity_norm</b>
    """
    story.append(Paragraph(formula_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Component Weights", heading_style))
    total_weight = sum(weights.values())
    
    weight_data = [['Component', 'Weight', 'Normalized Weight']]
    for key, value in weights.items():
        component_name = key.replace('w_', '').replace('_', ' ').title()
        normalized = (value / total_weight) if total_weight > 0 else 0
        weight_data.append([component_name, f"{value}", f"{normalized:.4f}"])
    
    weight_table = Table(weight_data, colWidths=[3*inch, 1*inch, 1.5*inch])
    weight_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(weight_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Normalization Methods", heading_style))
    story.append(Paragraph(
        "<b>Direct Normalization</b> (higher is better): value / max(value)",
        styles['Normal']
    ))
    story.append(Paragraph("Applied to: Revenue per Restaurant, % Sales Search", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Inverted Normalization</b> (lower represents opportunity): 1 - (value / max(value))",
        styles['Normal']
    ))
    story.append(Paragraph("Applied to: Reach Saturation, Digital Share, Ad Spend", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Reach Blend Method", heading_style))
    story.append(Paragraph(f"<b>Method Used:</b> {reach_method.upper()}", styles['Normal']))
    if reach_method == 'average':
        story.append(Paragraph("Reach Blend = (Meta Reach + TikTok Reach) / 2", styles['Normal']))
    elif reach_method == 'weighted':
        story.append(Paragraph("Reach Blend = Meta Weight × Meta Reach + (1 - Meta Weight) × TikTok Reach", styles['Normal']))
    elif reach_method == 'max':
        story.append(Paragraph("Reach Blend = MAX(Meta Reach, TikTok Reach)", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Normalization Max Values", heading_style))
    max_data = [['Metric', 'Max Value']]
    for key, value in max_values.items():
        metric_name = key.replace('_', ' ').title()
        max_data.append([metric_name, f"{value:.4f}"])
    
    max_table = Table(max_data, colWidths=[3.5*inch, 2*inch])
    max_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(max_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Opportunity Tier Thresholds", heading_style))
    if custom_thresholds:
        story.append(Paragraph("<b>Custom Thresholds Applied</b>", styles['Normal']))
    else:
        story.append(Paragraph("<b>Percentile-Based Thresholds</b>", styles['Normal']))
    
    threshold_data = [
        ['Tier', 'Threshold Range', 'MOI Value'],
        ['Lower', '< 33rd percentile', f'< {thresholds["p33"]:.4f}'],
        ['Moderate', '33rd - 66th percentile', f'{thresholds["p33"]:.4f} - {thresholds["p66"]:.4f}'],
        ['High', '66th - 99th percentile', f'{thresholds["p66"]:.4f} - {thresholds["p99"]:.4f}'],
        ['Exceptional', '≥ 99th percentile', f'≥ {thresholds["p99"]:.4f}']
    ]
    
    threshold_table = Table(threshold_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch])
    threshold_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(threshold_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("MOI Index Calculation", heading_style))
    story.append(Paragraph(
        "The MOI Index scales the MOI score to a 0-100 range for easier interpretation:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>MOI Index = 100 × (MOI - min(MOI)) / (max(MOI) - min(MOI))</b>",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Data Processing Pipeline", heading_style))
    pipeline_steps = [
        "1. <b>CSV Upload</b>: User uploads market-level data",
        "2. <b>Column Mapping</b>: Automatic detection and manual mapping of required fields",
        "3. <b>Data Cleaning</b>: Parse currency ($), percentages (%), and reach rates (0-1)",
        "4. <b>Reach Blending</b>: Combine Meta and TikTok reach saturation",
        "5. <b>Normalization</b>: Apply direct/inverted normalization using max values",
        "6. <b>MOI Computation</b>: Calculate weighted composite score",
        "7. <b>MOI Index</b>: Scale to 0-100 range",
        "8. <b>Tier Assignment</b>: Bucket by percentiles into opportunity tiers"
    ]
    
    for step in pipeline_steps:
        story.append(Paragraph(step, styles['Normal']))
        story.append(Spacer(1, 0.05*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

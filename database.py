import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()


class MOISnapshot(Base):
    """Store historical MOI snapshots for comparison over time."""
    __tablename__ = 'moi_snapshots'
    
    id = Column(Integer, primary_key=True, index=True)
    snapshot_name = Column(String(255), nullable=False)
    analysis_type = Column(String(50), nullable=False)
    snapshot_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    grouping_key = Column(String(255), nullable=False)
    moi_score = Column(Float, nullable=False)
    moi_index = Column(Float, nullable=False)
    tier = Column(String(50), nullable=False)
    revenue_per_restaurant = Column(Float)
    pct_sales_search = Column(Float)
    pct_sales_google = Column(Float)
    ad_spend_per_restaurant = Column(Float)
    meta_reach = Column(Float)
    tiktok_reach = Column(Float)
    weights_config = Column(JSON)
    snapshot_metadata = Column(JSON)


def init_db():
    """Initialize database tables."""
    if engine:
        Base.metadata.create_all(bind=engine)


def save_moi_snapshot(
    snapshot_name: str,
    analysis_type: str,
    results_df: pd.DataFrame,
    grouping_col: str,
    weights: dict,
    reach_method: str
):
    """Save MOI results as a historical snapshot."""
    if not engine:
        raise Exception("Database not configured")
    
    init_db()
    
    db = SessionLocal()
    
    try:
        for _, row in results_df.iterrows():
            snapshot = MOISnapshot(
                snapshot_name=snapshot_name,
                analysis_type=analysis_type,
                snapshot_date=datetime.utcnow(),
                grouping_key=str(row[grouping_col]),
                moi_score=float(row['MOI']) if 'MOI' in row else 0.0,
                moi_index=float(row['MOI_Index']) if 'MOI_Index' in row else 0.0,
                tier=str(row['Tier']) if 'Tier' in row else 'N/A',
                revenue_per_restaurant=float(row['revenue_per_restaurant']) if 'revenue_per_restaurant' in row and pd.notna(row['revenue_per_restaurant']) else None,
                pct_sales_search=float(row['pct_sales_search']) if 'pct_sales_search' in row and pd.notna(row['pct_sales_search']) else None,
                pct_sales_google=float(row['pct_sales_google']) if 'pct_sales_google' in row and pd.notna(row['pct_sales_google']) else None,
                ad_spend_per_restaurant=float(row['ad_spend_per_restaurant']) if 'ad_spend_per_restaurant' in row and pd.notna(row['ad_spend_per_restaurant']) else None,
                meta_reach=float(row['meta_reach']) if 'meta_reach' in row and pd.notna(row['meta_reach']) else None,
                tiktok_reach=float(row['tiktok_reach']) if 'tiktok_reach' in row and pd.notna(row['tiktok_reach']) else None,
                weights_config=weights,
                snapshot_metadata={'reach_method': reach_method}
            )
            db.add(snapshot)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def get_snapshot_names():
    """Get list of all snapshot names."""
    if not engine:
        return []
    
    init_db()
    
    db = SessionLocal()
    try:
        snapshots = db.query(MOISnapshot.snapshot_name, MOISnapshot.snapshot_date, MOISnapshot.analysis_type).distinct().all()
        return [{'name': s[0], 'date': s[1], 'type': s[2]} for s in snapshots]
    finally:
        db.close()


def get_snapshot_data(snapshot_name: str):
    """Retrieve a specific snapshot's data."""
    if not engine:
        return None
    
    init_db()
    
    db = SessionLocal()
    try:
        snapshots = db.query(MOISnapshot).filter(MOISnapshot.snapshot_name == snapshot_name).all()
        
        if not snapshots:
            return None
        
        data = []
        for s in snapshots:
            data.append({
                'grouping_key': s.grouping_key,
                'MOI': s.moi_score,
                'MOI_Index': s.moi_index,
                'Tier': s.tier,
                'revenue_per_restaurant': s.revenue_per_restaurant,
                'pct_sales_search': s.pct_sales_search,
                'pct_sales_google': s.pct_sales_google,
                'ad_spend_per_restaurant': s.ad_spend_per_restaurant,
                'meta_reach': s.meta_reach,
                'tiktok_reach': s.tiktok_reach,
                'snapshot_date': s.snapshot_date
            })
        
        return pd.DataFrame(data)
    finally:
        db.close()


def compare_snapshots(snapshot1_name: str, snapshot2_name: str):
    """Compare two snapshots and return MOI changes."""
    df1 = get_snapshot_data(snapshot1_name)
    df2 = get_snapshot_data(snapshot2_name)
    
    if df1 is None or df2 is None:
        return None
    
    merged = pd.merge(
        df1[['grouping_key', 'MOI', 'MOI_Index', 'Tier']],
        df2[['grouping_key', 'MOI', 'MOI_Index', 'Tier']],
        on='grouping_key',
        how='outer',
        suffixes=('_baseline', '_current')
    )
    
    merged['MOI_Change'] = merged['MOI_current'] - merged['MOI_baseline']
    merged['MOI_Index_Change'] = merged['MOI_Index_current'] - merged['MOI_Index_baseline']
    
    tier_order = {'Lower': 1, 'Moderate': 2, 'High': 3, 'Exceptional': 4, 'N/A': 0}
    
    def classify_tier_change(row):
        if pd.isna(row['Tier_current']) or pd.isna(row['Tier_baseline']):
            return 'N/A'
        
        baseline_score = tier_order.get(row['Tier_baseline'], 0)
        current_score = tier_order.get(row['Tier_current'], 0)
        
        if current_score > baseline_score:
            return 'Improved'
        elif current_score < baseline_score:
            return 'Declined'
        else:
            return 'Same'
    
    merged['Tier_Change'] = merged.apply(classify_tier_change, axis=1)
    
    return merged


def delete_snapshot(snapshot_name: str):
    """Delete a snapshot."""
    if not engine:
        return False
    
    init_db()
    
    db = SessionLocal()
    try:
        db.query(MOISnapshot).filter(MOISnapshot.snapshot_name == snapshot_name).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

import unittest
import pandas as pd
import numpy as np
import moi_utils as moi


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functions."""
    
    def test_parse_currency(self):
        """Test currency parsing."""
        self.assertEqual(moi.parse_currency("$1,234.56"), 1234.56)
        self.assertEqual(moi.parse_currency("$100"), 100.0)
        self.assertEqual(moi.parse_currency(500), 500.0)
        self.assertTrue(np.isnan(moi.parse_currency(None)))
        self.assertTrue(np.isnan(moi.parse_currency("invalid")))
    
    def test_parse_percentage(self):
        """Test percentage parsing."""
        self.assertAlmostEqual(moi.parse_percentage("25%"), 0.25)
        self.assertAlmostEqual(moi.parse_percentage("50"), 0.50)
        self.assertAlmostEqual(moi.parse_percentage(0.75), 0.75)
        self.assertAlmostEqual(moi.parse_percentage(80), 0.80)
        self.assertTrue(np.isnan(moi.parse_percentage(None)))
    
    def test_normalize_reach_rate(self):
        """Test reach rate normalization."""
        self.assertEqual(moi.normalize_reach_rate(0.5), 0.5)
        self.assertEqual(moi.normalize_reach_rate(75), 0.75)
        self.assertEqual(moi.normalize_reach_rate(1.0), 1.0)
        self.assertTrue(np.isnan(moi.normalize_reach_rate(None)))


class TestColumnAutoDetection(unittest.TestCase):
    """Test column auto-detection."""
    
    def test_auto_detect_revenue(self):
        """Test auto-detection of revenue column."""
        columns = ['DMA', 'Revenue per Store', 'Sales']
        result = moi.auto_detect_column(columns, 'revenue_per_restaurant')
        self.assertEqual(result, 'Revenue per Store')
    
    def test_auto_detect_dma(self):
        """Test auto-detection of DMA column."""
        columns = ['DMA Region', 'County', 'Sales']
        result = moi.auto_detect_column(columns, 'grouping_dma')
        self.assertEqual(result, 'DMA Region')
    
    def test_auto_detect_not_found(self):
        """Test auto-detection when column not found."""
        columns = ['Unknown1', 'Unknown2']
        result = moi.auto_detect_column(columns, 'revenue_per_restaurant')
        self.assertIsNone(result)


class TestReachBlend(unittest.TestCase):
    """Test reach blend computation."""
    
    def setUp(self):
        """Set up test data."""
        self.meta = pd.Series([0.5, 0.6, 0.7])
        self.tiktok = pd.Series([0.4, 0.5, 0.8])
    
    def test_average_blend(self):
        """Test average blend method."""
        result = moi.compute_reach_blend(self.meta, self.tiktok, method='average')
        expected = pd.Series([0.45, 0.55, 0.75])
        pd.testing.assert_series_equal(result, expected)
    
    def test_weighted_blend(self):
        """Test weighted blend method."""
        result = moi.compute_reach_blend(
            self.meta, self.tiktok, method='weighted', meta_weight=0.6
        )
        expected = pd.Series([0.46, 0.56, 0.74])
        pd.testing.assert_series_equal(result, expected)
    
    def test_max_blend(self):
        """Test max blend method."""
        result = moi.compute_reach_blend(self.meta, self.tiktok, method='max')
        expected = pd.Series([0.5, 0.6, 0.8])
        pd.testing.assert_series_equal(result, expected)


class TestNormalization(unittest.TestCase):
    """Test normalization functions."""
    
    def test_normalize_direct(self):
        """Test direct normalization (higher is better)."""
        series = pd.Series([100, 200, 300, 400])
        normalized, max_val, warnings = moi.normalize_direct(series)
        
        self.assertEqual(max_val, 400)
        self.assertEqual(len(warnings), 0)
        self.assertAlmostEqual(normalized.iloc[0], 0.25)
        self.assertAlmostEqual(normalized.iloc[3], 1.0)
    
    def test_normalize_inverted(self):
        """Test inverted normalization (lower is better)."""
        series = pd.Series([10, 20, 30, 40])
        normalized, max_val, warnings = moi.normalize_inverted(series)
        
        self.assertEqual(max_val, 40)
        self.assertEqual(len(warnings), 0)
        self.assertAlmostEqual(normalized.iloc[0], 0.75)
        self.assertAlmostEqual(normalized.iloc[3], 0.0)
    
    def test_normalize_all_zeros(self):
        """Test normalization with all zero values."""
        series = pd.Series([0, 0, 0])
        normalized, max_val, warnings = moi.normalize_direct(series)
        
        self.assertEqual(max_val, 0)
        self.assertGreater(len(warnings), 0)
        self.assertEqual(normalized.sum(), 0)
    
    def test_normalize_with_nan(self):
        """Test normalization with NaN values."""
        series = pd.Series([100, np.nan, 200, 300])
        normalized, max_val, warnings = moi.normalize_direct(series)
        
        self.assertEqual(max_val, 300)
        self.assertEqual(normalized.iloc[1], 0)


class TestMOIComputation(unittest.TestCase):
    """Test MOI computation."""
    
    def test_compute_moi_equal_weights(self):
        """Test MOI computation with equal weights."""
        components = {
            'comp1': pd.Series([0.5, 0.6, 0.7]),
            'comp2': pd.Series([0.4, 0.5, 0.8])
        }
        weights = {
            'comp1': 50,
            'comp2': 50
        }
        
        moi_result = moi.compute_moi(components, weights)
        
        self.assertAlmostEqual(moi_result.iloc[0], 0.45)
        self.assertAlmostEqual(moi_result.iloc[1], 0.55)
        self.assertAlmostEqual(moi_result.iloc[2], 0.75)
    
    def test_compute_moi_weighted(self):
        """Test MOI computation with custom weights."""
        components = {
            'comp1': pd.Series([1.0, 0.5]),
            'comp2': pd.Series([0.0, 1.0])
        }
        weights = {
            'comp1': 75,
            'comp2': 25
        }
        
        moi_result = moi.compute_moi(components, weights)
        
        self.assertAlmostEqual(moi_result.iloc[0], 0.75)
        self.assertAlmostEqual(moi_result.iloc[1], 0.625)


class TestOpportunityTiers(unittest.TestCase):
    """Test opportunity tier assignment."""
    
    def test_assign_tiers_normal(self):
        """Test tier assignment with normal distribution."""
        moi_series = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
        tiers, thresholds = moi.assign_opportunity_tiers(moi_series)
        
        self.assertIn('p33', thresholds)
        self.assertIn('p66', thresholds)
        self.assertIn('p99', thresholds)
        
        self.assertEqual(tiers.iloc[0], 'Lower')
        self.assertIn(tiers.iloc[-1], ['High', 'Exceptional'])
    
    def test_assign_tiers_small_dataset(self):
        """Test tier assignment with small dataset."""
        moi_series = pd.Series([0.5, 0.9])
        tiers, thresholds = moi.assign_opportunity_tiers(moi_series)
        
        self.assertEqual(thresholds['p33'], 0.65)
        self.assertEqual(thresholds['p66'], 0.80)
        self.assertEqual(thresholds['p99'], 0.95)
    
    def test_assign_tiers_with_nan(self):
        """Test tier assignment with NaN values."""
        moi_series = pd.Series([0.1, np.nan, 0.5, 0.9])
        tiers, thresholds = moi.assign_opportunity_tiers(moi_series)
        
        self.assertEqual(tiers.iloc[1], 'N/A')


class TestMOIIndex(unittest.TestCase):
    """Test MOI Index computation."""
    
    def test_moi_index_calculation(self):
        """Test MOI Index scaling to 0-100."""
        moi_series = pd.Series([0.0, 0.5, 1.0])
        index = moi.compute_moi_index(moi_series)
        
        self.assertEqual(index.iloc[0], 0)
        self.assertEqual(index.iloc[1], 50)
        self.assertEqual(index.iloc[2], 100)
    
    def test_moi_index_constant(self):
        """Test MOI Index with constant values."""
        moi_series = pd.Series([0.5, 0.5, 0.5])
        index = moi.compute_moi_index(moi_series)
        
        self.assertTrue(all(index == 50))


class TestCountyDMARollup(unittest.TestCase):
    """Test county to DMA aggregation."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'County': ['County1', 'County2', 'County3'],
            'DMA': ['DMA1', 'DMA1', 'DMA2'],
            'revenue': [100, 200, 300],
            'pct_search': [0.1, 0.2, 0.3],
            'pct_google': [0.4, 0.5, 0.6],
            'ad_spend': [50, 75, 100],
            'meta_reach': [0.5, 0.6, 0.7],
            'tiktok_reach': [0.4, 0.5, 0.6]
        })
    
    def test_aggregate_median(self):
        """Test aggregation with median."""
        result = moi.aggregate_county_to_dma(
            self.df,
            county_col='County',
            dma_col='DMA',
            revenue_col='revenue',
            pct_search_col='pct_search',
            pct_google_col='pct_google',
            ad_spend_col='ad_spend',
            meta_reach_col='meta_reach',
            tiktok_reach_col='tiktok_reach',
            revenue_agg='median',
            spend_agg='median'
        )
        
        self.assertEqual(len(result), 2)
        self.assertIn('DMA', result.columns)
        self.assertIn('revenue', result.columns)
    
    def test_aggregate_mean(self):
        """Test aggregation with mean."""
        result = moi.aggregate_county_to_dma(
            self.df,
            county_col='County',
            dma_col='DMA',
            revenue_col='revenue',
            pct_search_col='pct_search',
            pct_google_col='pct_google',
            ad_spend_col='ad_spend',
            meta_reach_col='meta_reach',
            tiktok_reach_col='tiktok_reach',
            revenue_agg='mean',
            spend_agg='mean'
        )
        
        self.assertEqual(len(result), 2)
        dma1_revenue = result[result['DMA'] == 'DMA1']['revenue'].iloc[0]
        self.assertEqual(dma1_revenue, 150)


class TestExportHelpers(unittest.TestCase):
    """Test export helper functions."""
    
    def test_create_csv(self):
        """Test CSV creation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        csv_bytes = moi.create_download_csv(df)
        self.assertIsInstance(csv_bytes, bytes)
        self.assertGreater(len(csv_bytes), 0)
    
    def test_create_excel(self):
        """Test Excel creation."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        xlsx_bytes = moi.create_download_excel(df, 'TestSheet')
        self.assertIsInstance(xlsx_bytes, bytes)
        self.assertGreater(len(xlsx_bytes), 0)


if __name__ == '__main__':
    unittest.main()

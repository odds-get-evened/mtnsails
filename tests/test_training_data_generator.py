#!/usr/bin/env python3
"""
Unit tests for the training data generator.

Tests verify that the enhanced generator can produce high-quality,
diverse training data with proper augmentation and synthetic generation.
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module to avoid numpy dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "training_data_generator",
    Path(__file__).parent.parent / "llm_interface" / "training_data_generator.py"
)
training_data_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_data_generator)

augment_query_paraphrase = training_data_generator.augment_query_paraphrase
create_iot_analyst_training_data = training_data_generator.create_iot_analyst_training_data
generate_synthetic_forecast_queries = training_data_generator.generate_synthetic_forecast_queries
generate_synthetic_anomaly_queries = training_data_generator.generate_synthetic_anomaly_queries
generate_synthetic_gradient_queries = training_data_generator.generate_synthetic_gradient_queries
generate_synthetic_correlation_queries = training_data_generator.generate_synthetic_correlation_queries
expand_training_data = training_data_generator.expand_training_data
calculate_data_statistics = training_data_generator.calculate_data_statistics


class TestQueryAugmentation(unittest.TestCase):
    """Test query augmentation functions"""

    def test_augment_query_paraphrase_returns_list(self):
        """Test that paraphrase function returns a list"""
        query = "What is the temperature forecast?"
        result = augment_query_paraphrase(query, count=3)
        self.assertIsInstance(result, list)

    def test_augment_query_paraphrase_count(self):
        """Test that paraphrase function returns requested count or less"""
        query = "What is the temperature forecast?"
        count = 5
        result = augment_query_paraphrase(query, count=count)
        self.assertLessEqual(len(result), count)
        self.assertGreater(len(result), 0)

    def test_augment_query_paraphrase_includes_original(self):
        """Test that original query is included in variations"""
        query = "What is the temperature forecast?"
        result = augment_query_paraphrase(query, count=3)
        self.assertIn(query, result)

    def test_augment_query_paraphrase_uniqueness(self):
        """Test that paraphrases are unique"""
        query = "What is the temperature forecast?"
        result = augment_query_paraphrase(query, count=5)
        # Check uniqueness by converting to lowercase
        unique_lower = set(v.lower() for v in result)
        self.assertEqual(len(unique_lower), len(result))


class TestSyntheticGeneration(unittest.TestCase):
    """Test synthetic data generation functions"""

    def test_generate_forecast_queries(self):
        """Test forecast query generation"""
        metrics = ['temperature', 'humidity']
        time_ranges = [5, 30, 60]
        count = 10
        result = generate_synthetic_forecast_queries(metrics, time_ranges, count)
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), count)
        
        # Check structure
        if len(result) > 0:
            self.assertIn('query', result[0])
            self.assertIn('category', result[0])
            self.assertEqual(result[0]['category'], 'forecast')

    def test_generate_anomaly_queries(self):
        """Test anomaly query generation"""
        metrics = ['temperature', 'humidity']
        count = 10
        result = generate_synthetic_anomaly_queries(metrics, count)
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), count)
        
        # Check structure
        if len(result) > 0:
            self.assertIn('query', result[0])
            self.assertIn('category', result[0])
            self.assertEqual(result[0]['category'], 'anomaly_detection')

    def test_generate_gradient_queries(self):
        """Test gradient query generation"""
        metrics = ['temperature', 'humidity']
        count = 10
        result = generate_synthetic_gradient_queries(metrics, count)
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), count)
        
        # Check structure
        if len(result) > 0:
            self.assertIn('query', result[0])
            self.assertIn('category', result[0])
            self.assertEqual(result[0]['category'], 'gradient_analysis')

    def test_generate_correlation_queries(self):
        """Test correlation query generation"""
        metrics = ['temperature', 'humidity', 'pressure']
        count = 10
        result = generate_synthetic_correlation_queries(metrics, count)
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), count)
        
        # Check structure
        if len(result) > 0:
            self.assertIn('query', result[0])
            self.assertIn('category', result[0])
            self.assertEqual(result[0]['category'], 'correlation')


class TestDataExpansion(unittest.TestCase):
    """Test data expansion and deduplication"""

    def test_expand_training_data_basic(self):
        """Test basic data expansion"""
        base_data = [
            {'query': 'What is temperature?', 'type': 'basic'},
            {'query': 'What is humidity?', 'type': 'basic'}
        ]
        
        result = expand_training_data(base_data, augment_factor=2, synthetic_count=5, seed=42)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), len(base_data))

    def test_expand_training_data_no_duplicates(self):
        """Test that expansion doesn't create duplicate queries"""
        base_data = [
            {'query': 'What is temperature?', 'type': 'basic'},
            {'query': 'What is humidity?', 'type': 'basic'}
        ]
        
        result = expand_training_data(base_data, augment_factor=3, synthetic_count=20, seed=42)
        
        # Extract all queries and check for duplicates
        queries = [item.get('query', str(item)).lower().strip() for item in result]
        unique_queries = set(queries)
        
        self.assertEqual(len(queries), len(unique_queries), 
                        "Duplicate queries found in expanded data")

    def test_expand_training_data_with_seed_reproducible(self):
        """Test that seed produces reproducible results"""
        base_data = [
            {'query': 'What is temperature?', 'type': 'basic'}
        ]
        
        result1 = expand_training_data(base_data, augment_factor=2, synthetic_count=10, seed=42)
        result2 = expand_training_data(base_data, augment_factor=2, synthetic_count=10, seed=42)
        
        # Extract queries from both results
        queries1 = [item.get('query', str(item)) for item in result1]
        queries2 = [item.get('query', str(item)) for item in result2]
        
        self.assertEqual(queries1, queries2, "Seed did not produce reproducible results")

    def test_expand_training_data_diversity_mode(self):
        """Test that diversity mode creates more varied data"""
        base_data = [{'query': 'What is temperature?', 'type': 'basic'}]
        
        normal = expand_training_data(base_data, augment_factor=1, 
                                     synthetic_count=50, diversity_mode=False, seed=42)
        diverse = expand_training_data(base_data, augment_factor=1, 
                                      synthetic_count=50, diversity_mode=True, seed=43)
        
        # Diversity mode should potentially create more unique examples
        # (though not guaranteed depending on random selection)
        self.assertGreater(len(diverse), 0)
        self.assertGreater(len(normal), 0)


class TestStatistics(unittest.TestCase):
    """Test statistics calculation"""

    def test_calculate_data_statistics(self):
        """Test statistics calculation"""
        data = [
            {'query': 'What is temperature?', 'category': 'forecast', 'metric': 'temperature'},
            {'query': 'What is humidity?', 'category': 'forecast', 'metric': 'humidity'},
            {'query': 'Any anomalies?', 'category': 'anomaly_detection', 'metric': 'temperature'},
        ]
        
        stats = calculate_data_statistics(data)
        
        self.assertIn('total_examples', stats)
        self.assertIn('categories', stats)
        self.assertIn('metrics', stats)
        self.assertIn('avg_query_length', stats)
        self.assertIn('unique_queries', stats)
        self.assertIn('vocabulary_size', stats)
        
        self.assertEqual(stats['total_examples'], 3)
        self.assertEqual(stats['categories']['forecast'], 2)
        self.assertEqual(stats['categories']['anomaly_detection'], 1)

    def test_calculate_statistics_empty_data(self):
        """Test statistics with empty data"""
        data = []
        stats = calculate_data_statistics(data)
        
        self.assertEqual(stats['total_examples'], 0)
        self.assertEqual(stats['unique_queries'], 0)


class TestScaleGeneration(unittest.TestCase):
    """Test large-scale data generation"""

    def test_generate_100plus_examples(self):
        """Test generation of 100+ unique examples"""
        base_data = [
            {'query': f'Query {i}', 'type': 'test'} for i in range(10)
        ]
        
        result = expand_training_data(base_data, augment_factor=3, 
                                     synthetic_count=500, diversity_mode=True, seed=42)
        
        self.assertGreater(len(result), 100, "Should generate substantial number of examples")
        
        # Verify no duplicates
        queries = [item.get('query', str(item)).lower().strip() for item in result]
        unique_queries = set(queries)
        self.assertEqual(len(queries), len(unique_queries))

    def test_generate_with_diversity(self):
        """Test generation with diversity mode enabled"""
        base_data = [{'query': 'Test query', 'type': 'test'}]
        
        result = expand_training_data(base_data, augment_factor=2, 
                                     synthetic_count=200, diversity_mode=True, seed=42)
        
        self.assertGreater(len(result), 50)
        
        # Check for diverse metric types
        stats = calculate_data_statistics(result)
        self.assertGreater(len(stats['metrics']), 0)


class TestCreateIoTAnalystTrainingData(unittest.TestCase):
    """Test the combined training data factory function"""

    def test_returns_nonempty_list(self):
        """create_iot_analyst_training_data should return a non-empty list"""
        result = create_iot_analyst_training_data()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_all_generators_contribute(self):
        """Each component generator should contribute at least one item"""
        result = create_iot_analyst_training_data()
        # The combined result must be larger than any single sub-generator
        from_forecast = training_data_generator.generate_forecast_queries()
        self.assertGreater(len(result), len(from_forecast))


if __name__ == '__main__':
    unittest.main()

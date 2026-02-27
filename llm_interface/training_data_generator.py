"""
Training Data Generator for MTN Sails LLM
"""  

import json
import argparse
import random
import itertools
import re
from typing import Any, Dict, List, Set


def generate_forecast_queries() -> List[Dict[str, Any]]:
    # Create parameterized forecast variations  
    variations = []  
    for duration in [1, 3, 5]:
        for query_template in ["What is the forecast for temperature in {duration} hours?",  
                                "What will be the humidity in {duration} hours?"]:
            variations.append({"query": query_template.format(duration=duration), "duration": duration})
    return variations


def generate_forecast_results_variations() -> List[Dict[str, Any]]:
    # Generate realistic LSTM forecast result interpretations with trends  
    results = []
    for trend in ["rising", "falling", "stable"]:
        results.append({"trend": trend, "result": random.uniform(20.0, 40.0)})
    return results


def generate_anomaly_scenarios() -> List[Dict[str, Any]]:
    # Create comprehensive anomaly examples
    anomalies = []
    for severity in ["Low", "Medium", "High", "Critical"]:
        for anomaly_type in ["Temperature", "Humidity", "Pressure"]:
            anomalies.append({"severity": severity, "type": anomaly_type})
    return anomalies


def get_action_for_anomaly_type(anomaly_type: str) -> str:
    # Provide specific recommendations based on the anomaly type
    actions = {
        "Temperature": "Check HVAC system.",
        "Humidity": "Ensure ventilation.",
        "Pressure": "Calibrate instruments."
    }
    return actions.get(anomaly_type, "No action defined.")


def generate_gradient_analysis() -> List[Dict[str, Any]]:
    # Generate spatial gradient analysis queries
    gradients = []
    for point in range(1, 6):
        gradients.append({"point": point, "analysis": "Analyze gradient at point {}".format(point)})
    return gradients


def generate_realistic_sensor_scenarios() -> List[Dict[str, Any]]:
    # Create time-of-day patterns
    scenarios = []
    for time in range(0, 24):
        scenarios.append({"time": time, "scenario": "Sensor reading at {}:00".format(time)})
    return scenarios


def generate_multi_sensor_correlations() -> List[Dict[str, Any]]:
    # Explain physical correlations between sensors
    correlations = []
    for sensor1, sensor2 in itertools.combinations(["Temperature", "Humidity", "Pressure"], 2):
        correlations.append({"correlation": f"{sensor1} correlates with {sensor2}."})
    return correlations


def generate_domain_knowledge() -> List[Dict[str, Any]]:
    # Create domain knowledge Q&A
    knowledge = []
    questions = ["What affects temperature?"]
    for question in questions:
        knowledge.append({"question": question, "answer": "Temperature is affected by weather conditions."})
    return knowledge


def generate_combinatorial_queries() -> List[Dict[str, Any]]:
    # Using all component combinations
    combinatorial = []
    for query in itertools.product(["What is", "How is"], ["temperature", "humidity"]):
        combinatorial.append({"query": f"{query[0]} the {query[1]}?"})
    return combinatorial


def add_variations(data: str) -> List[str]:
    # Data augmentation with typos, abbreviations, paraphrasing
    variations = [data, data.replace("temperature", "temp"), data + "!", data.lower()]
    return variations


def augment_query_paraphrase(query: str, count: int = 3) -> List[str]:
    """Generate paraphrased variations of a query using synonym replacement and restructuring"""
    variations = [query]
    
    # Synonym mappings for common terms
    synonyms = {
        "what is": ['what\'s', 'can you tell me', 'show me', 'display'],
        "what will be": ['what\'s going to be', 'predict', 'forecast', 'what do you expect for'],
        "how is": ['what\'s', 'tell me about', 'show me'],
        "forecast": ['prediction', 'projection', 'outlook', 'estimate'],
        "temperature": ['temp', 'temperature reading', 'thermal reading'],
        "humidity": ['moisture', 'humidity level', 'moisture level'],
        "in": ['for', 'over', 'within', 'during the next'],
        "hours": ['hrs', 'hours from now', 'hour period'],
        "analyze": ['examine', 'review', 'evaluate', 'assess'],
        "sensor": ['device', 'monitor', 'detector', 'probe'],
        "reading": ['value', 'measurement', 'data point', 'sample'],
    }
    
    # Generate variations by applying synonym replacements
    for _ in range(count - 1):
        variant = query.lower()
        # Randomly select and apply substitutions
        applied = False
        for original, replacements in synonyms.items():
            if original in variant and random.random() < 0.4:  # 40% chance to replace
                replacement = random.choice(replacements)
                variant = variant.replace(original, replacement, 1)
                applied = True
        
        # Add capitalization variations
        if applied:
            if random.random() < 0.5:
                variant = variant.capitalize()
            variations.append(variant)
    
    # Return unique variations maintaining order
    seen = set()
    unique_variations = []
    for v in variations:
        v_lower = v.lower()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variations.append(v)
    
    return unique_variations[:count]


def generate_synthetic_forecast_queries(metrics: List[str], time_ranges: List[int], count: int) -> List[Dict]:
    """Generate synthetic forecast queries using templates"""
    templates = [
        "What will {metric} be in {time} {unit}?",
        "Forecast {metric} for the next {time} {unit}",
        "Predict {metric} over {time} {unit}",
        "What is the {time} {unit} forecast for {metric}?",
        "Show me the {metric} prediction for {time} {unit}",
        "Can you forecast {metric} {time} {unit} ahead?",
        "What's the {metric} outlook for {time} {unit}?",
        "Estimate {metric} in {time} {unit}",
        "Tell me the {metric} forecast for {time} {unit}",
        "Give me a {time} {unit} {metric} prediction",
        "What do you predict for {metric} in {time} {unit}?",
        "Project {metric} for {time} {unit}",
        "What will happen to {metric} in {time} {unit}?",
        "Expected {metric} in {time} {unit}",
        "{metric} forecast: {time} {unit}",
        "Anticipate {metric} for {time} {unit}",
        "Get {metric} projection for {time} {unit}",
        "Future {metric} in {time} {unit}?",
        "Predict future {metric} over {time} {unit}",
        "{time} {unit} {metric} outlook",
    ]
    
    queries = []
    attempts = 0
    # Allow up to 50 attempts per desired query to find unique combinations
    # This ensures we can generate large datasets while avoiding infinite loops
    max_attempts = count * 50
    seen_queries_lower = set()
    
    while len(queries) < count and attempts < max_attempts:
        metric = random.choice(metrics)
        time_val = random.choice(time_ranges)
        template = random.choice(templates)
        
        # Convert time to appropriate unit
        if time_val < 60:
            time_str = str(time_val)
            unit = 'minutes' if time_val != 1 else 'minute'
        else:
            time_str = str(time_val // 60)
            unit = 'hours' if time_val // 60 != 1 else 'hour'
        
        query = template.format(metric=metric, time=time_str, unit=unit)
        query_lower = query.lower()
        
        # Create training example with metadata
        example = {
            'query': query,
            'category': 'forecast',
            'metric': metric,
            'time_range': time_val,
            'expected_action': f'Run LSTM forecast for {metric} with {time_val} minute prediction window'
        }
        
        # Avoid near-duplicates
        if query_lower not in seen_queries_lower:
            queries.append(example)
            seen_queries_lower.add(query_lower)
        
        attempts += 1
    
    return queries


def generate_synthetic_anomaly_queries(metrics: List[str], count: int) -> List[Dict]:
    """Generate synthetic anomaly detection queries"""
    templates = [
        "Are there any anomalies in {metric}?",
        "Detect anomalies in {metric} readings",
        "Check {metric} for unusual patterns",
        "Identify outliers in {metric} data",
        "Has {metric} shown any abnormal behavior?",
        "Find anomalies in the {metric} sensor",
        "Alert me to any {metric} anomalies",
        "Is {metric} behaving normally?",
        "Show me {metric} outliers",
        "Analyze {metric} for irregularities",
        "Are there unusual {metric} values?",
        "Detect {metric} abnormalities",
        "Find irregular {metric} patterns",
        "Check for {metric} outliers",
        "Spot {metric} anomalies",
        "Unusual {metric} behavior",
        "Any {metric} outliers?",
        "{metric} anomaly check",
        "Review {metric} for anomalies",
        "Scan {metric} data for outliers",
    ]
    
    queries = []
    attempts = 0
    max_attempts = count * 50
    seen_queries_lower = set()
    
    while len(queries) < count and attempts < max_attempts:
        metric = random.choice(metrics)
        template = random.choice(templates)
        query = template.format(metric=metric)
        query_lower = query.lower()
        
        example = {
            'query': query,
            'category': 'anomaly_detection',
            'metric': metric,
            'expected_action': f'Run anomaly detection on {metric} sensor data'
        }
        
        if query_lower not in seen_queries_lower:
            queries.append(example)
            seen_queries_lower.add(query_lower)
        
        attempts += 1
    
    return queries


def generate_synthetic_gradient_queries(metrics: List[str], count: int) -> List[Dict]:
    """Generate synthetic spatial gradient analysis queries"""
    templates = [
        "What is the {metric} gradient across sensors?",
        "Analyze spatial {metric} distribution",
        "Show me {metric} variation by location",
        "Compare {metric} between different points",
        "What's the {metric} differential across the space?",
        "Identify {metric} hotspots",
        "Map {metric} variations",
        "Show spatial {metric} patterns",
        "Detect {metric} gradients",
        "Where are the {metric} differences?",
        "Show {metric} spatial distribution",
        "Compare {metric} across locations",
        "{metric} gradient map",
        "Spatial {metric} analysis",
        "Location-based {metric} variance",
        "{metric} distribution patterns",
        "Identify {metric} zones",
        "Regional {metric} differences",
    ]
    
    queries = []
    attempts = 0
    max_attempts = count * 50
    seen_queries_lower = set()
    
    while len(queries) < count and attempts < max_attempts:
        metric = random.choice(metrics)
        template = random.choice(templates)
        query = template.format(metric=metric)
        query_lower = query.lower()
        
        example = {
            'query': query,
            'category': 'gradient_analysis',
            'metric': metric,
            'expected_action': f'Calculate spatial gradients for {metric} across sensor network'
        }
        
        if query_lower not in seen_queries_lower:
            queries.append(example)
            seen_queries_lower.add(query_lower)
        
        attempts += 1
    
    return queries


def generate_synthetic_correlation_queries(metrics: List[str], count: int) -> List[Dict]:
    """Generate synthetic correlation analysis queries"""
    queries = []
    attempts = 0
    max_attempts = count * 50
    seen_queries_lower = set()
    
    templates = [
        "How does {metric1} correlate with {metric2}?",
        "Show the relationship between {metric1} and {metric2}",
        "Is there a correlation between {metric1} and {metric2}?",
        "Analyze {metric1} vs {metric2}",
        "Compare {metric1} and {metric2} patterns",
        "What's the connection between {metric1} and {metric2}?",
        "Does {metric1} affect {metric2}?",
        "Show {metric1} and {metric2} correlation",
        "Relate {metric1} to {metric2}",
        "Compare patterns of {metric1} and {metric2}",
        "{metric1} vs {metric2} correlation",
        "Relationship: {metric1} and {metric2}",
        "{metric1}-{metric2} correlation analysis",
        "Link between {metric1} and {metric2}",
        "{metric1} impact on {metric2}",
        "Association of {metric1} with {metric2}",
    ]
    
    while len(queries) < count and attempts < max_attempts:
        if len(metrics) < 2:
            break
        metric1, metric2 = random.sample(metrics, 2)
        template = random.choice(templates)
        query = template.format(metric1=metric1, metric2=metric2)
        query_lower = query.lower()
        
        example = {
            'query': query,
            'category': 'correlation',
            'metrics': [metric1, metric2],
            'expected_action': f'Calculate correlation between {metric1} and {metric2}'
        }
        
        if query_lower not in seen_queries_lower:
            queries.append(example)
            seen_queries_lower.add(query_lower)
        
        attempts += 1
    
    return queries


def expand_training_data(base_data: List[Dict], augment_factor: int, synthetic_count: int, 
                        diversity_mode: bool = False, seed: int = None) -> List[Dict]:
    """Expand base training data using augmentation and synthesis"""
    if seed is not None:
        random.seed(seed)
    
    expanded_data = []
    seen_queries = set()
    
    # Augment existing data
    for item in base_data:
        # Add original
        expanded_data.append(item)
        query_key = item.get('query', str(item)).lower().strip()
        seen_queries.add(query_key)
        
        # Generate variations if item has a query field
        if 'query' in item and augment_factor > 1:
            variations = augment_query_paraphrase(item['query'], augment_factor)
            for variant in variations[1:]:  # Skip first (original)
                variant_key = variant.lower().strip()
                if variant_key not in seen_queries:
                    augmented = item.copy()
                    augmented['query'] = variant
                    expanded_data.append(augmented)
                    seen_queries.add(variant_key)
    
    # Generate synthetic data
    if synthetic_count > 0:
        metrics = ['temperature', 'humidity', 'light', 'barometer', 'vibration', 'pressure']
        
        # Add more metrics in diversity mode
        if diversity_mode:
            metrics.extend(['air quality', 'CO2', 'noise', 'motion', 'proximity', 
                          'soil moisture', 'wind speed', 'rain', 'UV index'])
        
        # Extended time ranges for diversity mode
        if diversity_mode:
            time_ranges = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 720, 1440]
        else:
            time_ranges = [5, 15, 30, 60, 120, 240, 360, 720, 1440]
        
        # Distribute synthetic data across categories
        forecast_count = int(synthetic_count * 0.4)
        anomaly_count = int(synthetic_count * 0.3)
        gradient_count = int(synthetic_count * 0.15)
        correlation_count = synthetic_count - forecast_count - anomaly_count - gradient_count
        
        # Generate each category
        synthetic_queries = []
        synthetic_queries.extend(generate_synthetic_forecast_queries(metrics, time_ranges, forecast_count))
        synthetic_queries.extend(generate_synthetic_anomaly_queries(metrics, anomaly_count))
        synthetic_queries.extend(generate_synthetic_gradient_queries(metrics, gradient_count))
        synthetic_queries.extend(generate_synthetic_correlation_queries(metrics, correlation_count))
        
        # Add synthetic data, avoiding duplicates
        for item in synthetic_queries:
            query_key = item.get('query', '').lower().strip()
            if query_key and query_key not in seen_queries:
                expanded_data.append(item)
                seen_queries.add(query_key)
    
    return expanded_data


def calculate_data_statistics(training_data: List[Dict]) -> Dict:
    """Calculate and return statistics about the generated data"""
    stats = {
        'total_examples': len(training_data),
        'categories': {},
        'metrics': {},
        'avg_query_length': 0,
        'unique_queries': 0,
        'vocabulary_size': 0,
    }
    
    # Collect all queries and words for analysis
    all_queries = []
    all_words = set()
    
    for item in training_data:
        # Category distribution
        category = item.get('category', 'other')
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # Metric distribution
        if 'metric' in item:
            metric = item['metric']
            stats['metrics'][metric] = stats['metrics'].get(metric, 0) + 1
        elif 'metrics' in item:
            for metric in item['metrics']:
                stats['metrics'][metric] = stats['metrics'].get(metric, 0) + 1
        
        # Query analysis
        if 'query' in item:
            query = item['query']
            all_queries.append(query)
            # Extract words for vocabulary
            words = re.findall(r'\b\w+\b', query.lower())
            all_words.update(words)
    
    # Calculate query statistics
    if all_queries:
        stats['avg_query_length'] = sum(len(q) for q in all_queries) / len(all_queries)
        stats['unique_queries'] = len(set(q.lower() for q in all_queries))
    
    stats['vocabulary_size'] = len(all_words)
    
    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    print(f"Total Examples: {stats['total_examples']}")
    print(f"Unique Queries: {stats['unique_queries']}")
    print(f"Vocabulary Size: {stats['vocabulary_size']} unique words")
    print(f"Average Query Length: {stats['avg_query_length']:.1f} characters")
    
    if stats['categories']:
        print("\nDistribution by Category:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_examples']) * 100
            print(f"  {category:20s}: {count:6d} ({percentage:5.1f}%)")
    
    if stats['metrics']:
        print("\nDistribution by Metric:")
        for metric, count in sorted(stats['metrics'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {metric:20s}: {count:6d}")
    
    print("="*60 + "\n")


def create_iot_analyst_training_data() -> List[Dict[str, Any]]:
    """Generate base IoT analyst training data combining all data sources"""
    return (generate_forecast_queries() + generate_forecast_results_variations() +
            generate_anomaly_scenarios() + generate_gradient_analysis() +
            generate_realistic_sensor_scenarios() + generate_multi_sensor_correlations() +
            generate_domain_knowledge() + generate_combinatorial_queries())


def main():
    # Comprehensive CLI using argparse
    parser = argparse.ArgumentParser(description="Training Data Generator for MTN Sails LLM")
    parser.add_argument('--output', type=str, default='iot_analyst_training.json', help='Output file')
    parser.add_argument('--count', type=int, help='Count of samples (legacy, use --total-target instead)')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON')
    parser.add_argument('--augment', type=int, default=2, help='Augmentations (legacy, use --augment-factor instead)')
    
    # New enhanced options
    parser.add_argument('--augment-factor', type=int, default=None, 
                       help='Generate N variations of each base example')
    parser.add_argument('--synthetic-count', type=int, default=0,
                       help='Generate N synthetic examples using templates')
    parser.add_argument('--total-target', type=int, default=None,
                       help='Generate data until reaching N total examples')
    parser.add_argument('--diversity-mode', action='store_true',
                       help='Enable maximum variation in generated data')
    parser.add_argument('--seed', type=int, default=None,
                       help='Set random seed for reproducible generation')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress informational output (only show statistics and file info)')
    
    args = parser.parse_args()  
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Resolve augment factor
    augment_factor = args.augment_factor if args.augment_factor is not None else args.augment
    
    # Generate base data
    base_data = generate_forecast_queries() + generate_forecast_results_variations() + \
                generate_anomaly_scenarios() + generate_gradient_analysis() + \
                generate_realistic_sensor_scenarios() + generate_multi_sensor_correlations() + \
                generate_domain_knowledge() + generate_combinatorial_queries()
    
    # Handle total-target mode
    if args.total_target:
        # Calculate how much synthetic data we need
        base_count = len(base_data)
        expected_after_augment = base_count * augment_factor
        synthetic_needed = max(0, args.total_target - expected_after_augment)
        
        if not args.quiet:
            print(f'Target: {args.total_target} examples')
            print(f'Base examples: {base_count}')
            print(f'After augmentation (factor {augment_factor}): ~{expected_after_augment}')
            print(f'Synthetic examples needed: {synthetic_needed}')
        
        data = expand_training_data(base_data, augment_factor, synthetic_needed, 
                                   args.diversity_mode, args.seed)
    else:
        # Standard mode with explicit augment and synthetic counts
        data = expand_training_data(base_data, augment_factor, args.synthetic_count,
                                   args.diversity_mode, args.seed)
    
    # Apply count limit if specified (legacy option)
    if args.count:
        data = data[:args.count]
    
    # Calculate and display statistics
    stats = calculate_data_statistics(data)
    print_statistics(stats)
    
    # Write output file
    if args.pretty:
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(args.output, 'w') as f:
            json.dump(data, f)
    
    print(f'✓ Output written to {args.output}')
    print(f'✓ File size: {len(data)} entries')

if __name__ == '__main__':
    main()
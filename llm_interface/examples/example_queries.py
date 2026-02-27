"""
Example Queries for MTN Sails + LSTM Bridge

Demonstrates various usage scenarios for natural language sensor queries.
"""
import sys
from pathlib import Path

# Add parent directories to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from the package
sys.path.insert(0, str(Path(__file__).parent.parent))
from mtnsails_bridge import MTNSailsLSTMBridge


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def example_temperature_forecast(bridge: MTNSailsLSTMBridge):
    """Example: Temperature forecasting"""
    print_section("Example 1: Temperature Forecast")
    
    queries = [
        "What will the temperature be in 2 hours?",
        "Show me temperature forecast for next hour",
        "Predict temperature for the next 30 minutes"
    ]
    
    for query in queries:
        print(f"💬 Query: {query}")
        response = bridge.ask(query)
        print(f"🤖 Response:\n{response}\n")
        print("-" * 70 + "\n")


def example_anomaly_detection(bridge: MTNSailsLSTMBridge):
    """Example: Anomaly detection"""
    print_section("Example 2: Anomaly Detection")
    
    queries = [
        "Are there any unusual temperature readings?",
        "Is the pump going to fail soon?",
        "Check for sensor failures"
    ]
    
    for query in queries:
        print(f"💬 Query: {query}")
        response = bridge.ask(query)
        print(f"🤖 Response:\n{response}\n")
        print("-" * 70 + "\n")


def example_gradient_analysis(bridge: MTNSailsLSTMBridge):
    """Example: Spatial gradient analysis"""
    print_section("Example 3: Gradient Analysis")
    
    queries = [
        "Which sensors have the biggest temperature difference?",
        "Show me humidity gradients between sensors",
        "Where are the largest light level differences?"
    ]
    
    for query in queries:
        print(f"💬 Query: {query}")
        response = bridge.ask(query)
        print(f"🤖 Response:\n{response}\n")
        print("-" * 70 + "\n")


def example_multi_turn_conversation(bridge: MTNSailsLSTMBridge):
    """Example: Multi-turn conversation"""
    print_section("Example 4: Multi-Turn Conversation")
    
    conversation = [
        "Show me temp forecast",
        "What will humidity be in the next hour?",
        "Are there any anomalies?",
        "Which sensors have the biggest temperature difference?"
    ]
    
    print("💭 Simulating a natural conversation flow:\n")
    
    for i, query in enumerate(conversation, 1):
        print(f"Turn {i}:")
        print(f"💬 User: {query}")
        response = bridge.ask(query)
        print(f"🤖 Assistant:\n{response}\n")
        print("-" * 70 + "\n")


def example_different_metrics(bridge: MTNSailsLSTMBridge):
    """Example: Different sensor metrics"""
    print_section("Example 5: Different Sensor Metrics")
    
    metrics = [
        ("Temperature", "What will temperature be in 1 hour?"),
        ("Humidity", "Predict humidity for next 2 hours"),
        ("Light", "Show me light level forecast"),
        ("Barometer", "What will barometric pressure be in 30 minutes?")
    ]
    
    for metric_name, query in metrics:
        print(f"📊 {metric_name}:")
        print(f"💬 Query: {query}")
        response = bridge.ask(query)
        print(f"🤖 Response:\n{response}\n")
        print("-" * 70 + "\n")


def main():
    """Run all examples"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run example queries for MTN Sails + LSTM bridge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model paths
  python example_queries.py --taber-model outputs/
  
  # Run with MTN Sails LLM
  python example_queries.py --mtnsails-model onnx_model --taber-model outputs/
  
  # Run specific example
  python example_queries.py --taber-model outputs/ --example forecast
        """
    )
    
    parser.add_argument(
        '--mtnsails-model',
        type=str,
        default=None,
        help='Path to MTN Sails ONNX model directory (optional)'
    )
    
    parser.add_argument(
        '--taber-model',
        type=str,
        default='outputs/',
        help='Path to Taber LSTM model directory (default: outputs/)'
    )
    
    parser.add_argument(
        '--example',
        type=str,
        choices=['forecast', 'anomaly', 'gradient', 'conversation', 'metrics', 'all'],
        default='all',
        help='Which example to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Initialize bridge
    print("\n🚀 Initializing MTN Sails + LSTM Bridge...")
    print(f"   MTN Sails Model: {args.mtnsails_model or 'Not provided (using rule-based)'}")
    print(f"   Taber Model: {args.taber_model}")
    
    try:
        bridge = MTNSailsLSTMBridge(
            mtnsails_model_path=args.mtnsails_model,
            taber_model_path=args.taber_model
        )
        print("✓ Bridge initialized successfully\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize bridge: {e}")
        print("\nMake sure you have:")
        print("  1. Trained LSTM model in the specified directory")
        print("  2. Sensor data available for predictions")
        print("  3. Required dependencies installed")
        return 1
    
    # Run examples
    examples = {
        'forecast': example_temperature_forecast,
        'anomaly': example_anomaly_detection,
        'gradient': example_gradient_analysis,
        'conversation': example_multi_turn_conversation,
        'metrics': example_different_metrics
    }
    
    if args.example == 'all':
        for name, func in examples.items():
            try:
                func(bridge)
            except Exception as e:
                print(f"❌ Error in {name} example: {e}\n")
    else:
        try:
            examples[args.example](bridge)
        except Exception as e:
            print(f"❌ Error: {e}\n")
            return 1
    
    print_section("Examples Complete")
    print("✓ All examples finished successfully")
    print("\nNext steps:")
    print("  • Try your own queries with --interactive mode")
    print("  • Train MTN Sails with training_data_generator.py")
    print("  • Integrate with your production system")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

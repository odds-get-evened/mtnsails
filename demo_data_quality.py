#!/usr/bin/env python3
"""
Demo script to show data quality validation in action.
This script doesn't require ML dependencies.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_handler import ConversationDataHandler


def main():
    """Demonstrate data quality validation."""
    
    print("="*80)
    print("Data Quality Validation Demo")
    print("="*80)
    
    # Test with good data
    print("\n1. Testing with HIGH QUALITY data (example_conversations.json):")
    print("-" * 80)
    
    handler = ConversationDataHandler()
    handler.load_from_json('example_conversations.json')
    
    report = handler.analyze_dataset_quality()
    print(f"Total conversations: {report['total_conversations']}")
    print(f"Valid conversations: {report['valid_conversations']}")
    print(f"Invalid conversations: {report['invalid_conversations']}")
    print(f"Quality score: {report['quality_score']:.1%}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  {rec}")
    
    # Test with bad data
    print("\n\n2. Testing with LOW QUALITY data (bad_quality_data.json):")
    print("-" * 80)
    
    handler2 = ConversationDataHandler()
    handler2.load_from_json('bad_quality_data.json')
    
    report2 = handler2.analyze_dataset_quality()
    print(f"Total conversations: {report2['total_conversations']}")
    print(f"Valid conversations: {report2['valid_conversations']}")
    print(f"Invalid conversations: {report2['invalid_conversations']}")
    print(f"Quality score: {report2['quality_score']:.1%}")
    
    if report2['issue_summary']:
        print("\nIssues detected:")
        for issue_type, count in report2['issue_summary'].items():
            if count > 0:
                print(f"  - {issue_type.replace('_', ' ').title()}: {count}")
    
    if report2['problematic_examples']:
        print("\nExample problematic conversations:")
        for example in report2['problematic_examples']:
            print(f"\n  Conversation #{example['index']}:")
            print(f"    Input: {example['input'][:80]}")
            print(f"    Output: {example['output'][:80]}")
            print(f"    Issues: {', '.join(example['issues'])}")
    
    print("\nRecommendations:")
    for rec in report2['recommendations']:
        print(f"  {rec}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Good data quality score: {report['quality_score']:.1%}")
    print(f"Bad data quality score:  {report2['quality_score']:.1%}")
    print("\nThe bad data contains:")
    print("  - Repetitive responses that echo the same phrases over and over")
    print("  - Empty responses")
    print("  - Gibberish (like 'http://')")
    print("  - Very short, meaningless responses")
    print("\nTraining on bad quality data will produce a model that generates")
    print("similar low-quality, repetitive, and nonsensical responses.")
    print("\nALWAYS validate your training data quality before fine-tuning!")
    print("="*80)
    print("\n" + "="*80)
    print("HOW TO VALIDATE YOUR OWN DATA")
    print("="*80)
    print("\nTo validate YOUR training data, run:")
    print("\n  python main.py validate --data-file <your_data.json>")
    print("\nExamples:")
    print("  python main.py validate --data-file my_conversations.json")
    print("  python main.py validate --data-file chat_logs.json")
    print("\nThe validator will:")
    print("  • Analyze all conversations in your file")
    print("  • Calculate a quality score (0-100%)")
    print("  • Show specific issues found")
    print("  • Provide recommendations")
    print("  • Exit with status 0 if quality is good, 1 if critical")
    print("="*80)


if __name__ == '__main__':
    main()

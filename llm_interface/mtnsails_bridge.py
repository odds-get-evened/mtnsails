"""
MTN Sails Bridge - Connect LLM with LSTM Predictor

This module provides a bridge between MTN Sails LLM and Taber LSTM predictor,
enabling natural language queries for sensor data analysis.
"""
import os
import sys
import re
import json
import argparse
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# Add parent directory to path for imports - must be before importing pipeline
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False
    print("WARNING: optimum and transformers not installed. LLM features will be limited.")
    print("Install with: pip install optimum[onnxruntime] transformers")

# Import predictor - this should work from parent directory
try:
    from pipeline.predictor import ONNXPredictor
    HAS_PREDICTOR = True
except ImportError as e:
    print(f"ERROR: Failed to import ONNXPredictor: {e}")
    HAS_PREDICTOR = False
    ONNXPredictor = None

# Import gradient analyzer - handle both package and direct imports
try:
    spec = importlib.util.spec_from_file_location(
        "gradient_analyzer",
        Path(__file__).parent.parent / "pipeline" / "gradient_analyzer.py"
    )
    if spec and spec.loader:
        gradient_module = importlib.util.module_from_spec(spec)
        # Add pipeline directory to path temporarily for gradient_analyzer's imports
        pipeline_dir = str(Path(__file__).parent.parent / "pipeline")
        sys.path.insert(0, pipeline_dir)
        try:
            spec.loader.exec_module(gradient_module)
            GradientAnalyzer = gradient_module.GradientAnalyzer
            HAS_GRADIENT_ANALYZER = True
        finally:
            # Remove pipeline dir from path to avoid conflicts
            if pipeline_dir in sys.path:
                sys.path.remove(pipeline_dir)
    else:
        raise ImportError("Could not load gradient_analyzer module")
except Exception as e:
    print(f"WARNING: Gradient analyzer import failed: {e}")
    HAS_GRADIENT_ANALYZER = False
    GradientAnalyzer = None


# Anomaly detection constants
# These represent baseline prediction-error statistics used to detect outliers.
# Units match the raw metric values (e.g. °C for temperature, % for humidity).
# In production, calibrate these per-metric using historical validation data.
ANOMALY_NORMAL_ERROR_MEAN = 0.5  # Expected mean absolute prediction error under normal conditions
ANOMALY_NORMAL_ERROR_STD = 0.3   # Standard deviation of prediction error under normal conditions
# Anomaly score reaches 1.0 when error = MEAN + SIGMA_THRESHOLD * STD (i.e. 3-sigma above mean)
ANOMALY_SIGMA_THRESHOLD = 3.0    # Number of standard deviations above mean that maps to score 1.0


class MTNSailsLSTMBridge:
    """
    Bridge between MTN Sails LLM and Taber LSTM predictor
    
    Enables natural language queries for sensor data analysis by:
    1. Using LLM to parse user queries into structured intents
    2. Calling LSTM predictor based on intent
    3. Using LLM to explain LSTM results in human language

    LLM is required for both parsing and explanation – there are no
    rule-based or template fallbacks.  Validated interactions (where
    intent parsing and LSTM both succeed) are optionally logged to a
    JSONL retrain buffer via :mod:`llm_interface.retrain_buffer`.
    """
    
    def __init__(self, mtnsails_model_path: Optional[str] = None, 
                 taber_model_path: str = "outputs/",
                 buffer_dir: Optional[str] = None):
        """
        Initialize the MTN Sails <-> LSTM bridge
        
        Args:
            mtnsails_model_path: Path to MTN Sails ONNX model directory.
                                 Required for LLM-based parsing and explanation.
            taber_model_path: Path to Taber LSTM model directory (required)
            buffer_dir: Optional path to JSONL retrain buffer directory.
                        When set, validated interactions are logged here
                        for continual learning via the retrain daemon.
        """
        self.taber_model_path = Path(taber_model_path)
        self.mtnsails_model_path = Path(mtnsails_model_path) if mtnsails_model_path else None
        self.buffer_dir = buffer_dir

        # Placeholders captured during _llm_parse_query for buffer logging
        self._last_intent_prompt: Optional[str] = None
        self._last_llm_raw_response: Optional[str] = None
        
        # Initialize LLM components
        self.llm_model = None
        self.tokenizer = None
        
        if self.mtnsails_model_path and HAS_OPTIMUM:
            try:
                print(f"Loading MTN Sails model from: {self.mtnsails_model_path}")
                self.llm_model = ORTModelForCausalLM.from_pretrained(
                    str(self.mtnsails_model_path)
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.mtnsails_model_path)
                )
                print("✓ MTN Sails LLM loaded successfully")
            except Exception as e:
                print(f"WARNING: Failed to load MTN Sails model: {e}")
                self.llm_model = None
        elif self.mtnsails_model_path:
            print("WARNING: optimum not installed, cannot load MTN Sails model")
            print("Install with: pip install optimum[onnxruntime] transformers")
        
        # Initialize LSTM predictor
        try:
            print(f"Loading Taber LSTM predictor from: {self.taber_model_path}")
            self.predictor = ONNXPredictor(str(self.taber_model_path))
            self.predictor.load_data()
            print("✓ Taber LSTM predictor loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Taber LSTM predictor: {e}")
        
        # Initialize gradient analyzer
        if HAS_GRADIENT_ANALYZER and GradientAnalyzer is not None:
            try:
                print(f"Loading gradient analyzer...")
                self.gradient_analyzer = GradientAnalyzer(str(self.taber_model_path))
                self.gradient_analyzer.load_data()
                print("✓ Gradient analyzer loaded successfully")
            except Exception as e:
                print(f"WARNING: Failed to load gradient analyzer: {e}")
                self.gradient_analyzer = None
        else:
            print("WARNING: Gradient analyzer not available")
            self.gradient_analyzer = None
    
    def ask(self, user_query: str) -> str:
        """
        Main entry point - process natural language query
        
        Args:
            user_query: Natural language question from user
            
        Returns:
            Human-readable response
        """
        try:
            # Reset per-request state used for buffer logging
            self._last_intent_prompt = None
            self._last_llm_raw_response = None

            # Step 1: Parse query to extract intent via LLM
            print(f"\n🔍 Processing query: {user_query}")
            intent = self._parse_query(user_query)
            print(f"📋 Extracted intent: {intent}")
            
            # Step 2: Call LSTM based on intent
            lstm_output = self._call_lstm(intent)
            print(f"📊 LSTM output received")

            # Step 3: Log validated interaction to retrain buffer (LLM + LSTM both succeeded)
            if self.buffer_dir and "error" not in lstm_output:
                self._log_to_buffer(user_query, intent, lstm_output)
            
            # Step 4: Generate LLM explanation
            response = self._explain_results(user_query, intent, lstm_output)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}\n\nPlease try rephrasing your question."
    
    def _parse_query(self, query: str) -> Dict:
        """
        Parse user query to extract structured intent using the LLM.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with: type, metric, duration, sensor_id

        Raises:
            RuntimeError: If the LLM model is not loaded.
        """
        # LLM is required – no rule-based fallback
        if self.llm_model is None:
            raise RuntimeError(
                "LLM model is required for query parsing. "
                "Provide --mtnsails-model with a valid model path, "
                "or install dependencies: pip install optimum[onnxruntime] transformers"
            )
        return self._llm_parse_query(query)
    
    def _llm_parse_query(self, query: str) -> Dict:
        """
        Use LLM to extract structured intent from query.
        
        Stores the prompt and raw LLM response on the instance so that
        ``_log_to_buffer`` can include them in the retrain record.
        
        Args:
            query: Natural language query
            
        Returns:
            Intent dictionary
        """
        prompt = f"""Analyze this IoT sensor query and extract the intent.

User Query: {query}

Extract:
Query Type: forecast | anomaly | gradient
Metric: temp | light | humidity | barometer
Duration: N minutes (for forecast/anomaly)
Sensor ID: optional

Respond in this exact format:
Query Type: <type>
Metric: <metric>
Duration: <minutes> minutes
"""
        # Capture prompt for buffer logging
        self._last_intent_prompt = prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Capture raw response for buffer logging
        self._last_llm_raw_response = response
        
        return self._extract_intent_from_response(response)
    
    def _extract_intent_from_response(self, llm_response: str) -> Dict:
        """
        Parse LLM output to get structured intent
        
        Args:
            llm_response: Raw LLM output text
            
        Returns:
            Intent dictionary
        """
        intent = {
            'type': 'forecast',
            'metric': 'temp',
            'duration': 60,
            'sensor_id': None
        }
        
        # Extract query type
        type_match = re.search(r'Query Type:\s*(forecast|anomaly|gradient)', llm_response, re.IGNORECASE)
        if type_match:
            intent['type'] = type_match.group(1).lower()
        
        # Extract metric
        metric_match = re.search(r'Metric:\s*(temp|temperature|light|humidity|barometer|vibration)', 
                                llm_response, re.IGNORECASE)
        if metric_match:
            metric = metric_match.group(1).lower()
            # Normalize temperature alias
            if metric == 'temperature':
                metric = 'temp'
            # vibration is not a direct LSTM metric; anomaly detection on any
            # metric can surface mechanical/pump issues as elevated prediction error.
            intent['metric'] = metric
        
        # Extract duration
        duration_match = re.search(r'Duration:\s*(\d+)\s*minutes?', llm_response, re.IGNORECASE)
        if duration_match:
            intent['duration'] = int(duration_match.group(1))
        
        return intent
    
    def _rule_based_parse_query(self, query: str) -> Dict:
        """
        Rule-based query parsing as fallback
        
        Args:
            query: Natural language query
            
        Returns:
            Intent dictionary
        """
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ['gradient', 'difference', 'compare', 'which sensor']):
            query_type = 'gradient'
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'fail', 'wrong', 'problem']):
            query_type = 'anomaly'
        else:
            query_type = 'forecast'
        
        # Determine metric
        if any(word in query_lower for word in ['temp', 'temperature']):
            metric = 'temp'
        elif any(word in query_lower for word in ['humid', 'moisture']):
            metric = 'humidity'
        elif any(word in query_lower for word in ['light', 'brightness']):
            metric = 'light'
        elif any(word in query_lower for word in ['pressure', 'barometer']):
            metric = 'barometer'
        else:
            metric = 'temp'  # Default
        
        # Extract duration (for forecast/anomaly)
        duration = 60  # Default 1 hour
        
        # Try to find time expressions
        hour_match = re.search(r'(\d+)\s*hour', query_lower)
        minute_match = re.search(r'(\d+)\s*min', query_lower)
        
        if hour_match:
            duration = int(hour_match.group(1)) * 60
        elif minute_match:
            duration = int(minute_match.group(1))
        
        return {
            'type': query_type,
            'metric': metric,
            'duration': duration,
            'sensor_id': None
        }
    
    def _call_lstm(self, intent: Dict) -> Dict:
        """
        Call appropriate LSTM function based on intent
        
        Args:
            intent: Parsed intent dictionary
            
        Returns:
            LSTM output dictionary
        """
        if intent['type'] == 'forecast':
            return self._call_lstm_forecast(intent)
        elif intent['type'] == 'anomaly':
            return self._call_lstm_anomaly_detection(intent)
        elif intent['type'] == 'gradient':
            return self._call_lstm_gradient_analysis(intent)
        else:
            raise ValueError(f"Unknown intent type: {intent['type']}")
    
    def _call_lstm_forecast(self, intent: Dict) -> Dict:
        """
        Call Taber LSTM predictor for time-series forecast
        
        Args:
            intent: Intent with metric and duration
            
        Returns:
            Forecast results
        """
        try:
            # Get forecast
            steps_ahead = intent['duration']  # minutes
            forecast = self.predictor.forecast(steps_ahead=steps_ahead)
            
            metric = intent['metric']
            
            # Extract relevant metric predictions
            if metric in forecast['predictions']:
                predictions = forecast['predictions'][metric]
                
                # Get current value
                current_value = None
                if self.predictor.buffer_data is not None and len(self.predictor.buffer_data) > 0:
                    current_value = self.predictor.buffer_data[metric].iloc[-1]
                
                return {
                    'type': 'forecast',
                    'metric': metric,
                    'current_value': current_value,
                    'predictions': predictions,
                    'steps_ahead': steps_ahead,
                    'timestamps': forecast.get('timestamps', [])
                }
            else:
                return {
                    'type': 'forecast',
                    'metric': metric,
                    'error': f'Metric {metric} not available in predictions'
                }
                
        except Exception as e:
            return {
                'type': 'forecast',
                'metric': intent['metric'],
                'error': str(e)
            }
    
    def _call_lstm_anomaly_detection(self, intent: Dict) -> Dict:
        """
        Use LSTM prediction error for anomaly detection
        
        Args:
            intent: Intent with metric
            
        Returns:
            Anomaly detection results
        """
        try:
            # Get recent predictions to compare with actuals
            forecast = self.predictor.forecast(steps_ahead=1)
            metric = intent['metric']
            
            if metric not in forecast['predictions']:
                return {
                    'type': 'anomaly',
                    'metric': metric,
                    'error': f'Metric {metric} not available'
                }
            
            # Get prediction and actual
            prediction = forecast['predictions'][metric][0]
            
            # Get actual current value
            actual = None
            if self.predictor.buffer_data is not None and len(self.predictor.buffer_data) > 0:
                actual = self.predictor.buffer_data[metric].iloc[-1]
            
            if actual is None:
                return {
                    'type': 'anomaly',
                    'metric': metric,
                    'error': 'No actual data available for comparison'
                }
            
            # Calculate prediction error
            error = abs(actual - prediction)
            
            # Use constants for anomaly detection thresholds
            # Note: In production, these should come from validation data for each metric
            normal_error_mean = ANOMALY_NORMAL_ERROR_MEAN
            normal_error_std = ANOMALY_NORMAL_ERROR_STD
            
            # Calculate anomaly score using sigma threshold
            anomaly_score = min(1.0, max(0.0, 
                (error - normal_error_mean) / (ANOMALY_SIGMA_THRESHOLD * normal_error_std)))
            
            severity = 'Low'
            if anomaly_score > 0.7:
                severity = 'High'
            elif anomaly_score > 0.4:
                severity = 'Medium'
            
            return {
                'type': 'anomaly',
                'metric': metric,
                'prediction': prediction,
                'actual': actual,
                'error': error,
                'normal_error_range': (normal_error_mean - normal_error_std, 
                                     normal_error_mean + normal_error_std),
                'anomaly_score': anomaly_score,
                'severity': severity
            }
            
        except Exception as e:
            return {
                'type': 'anomaly',
                'metric': intent['metric'],
                'error': str(e)
            }
    
    def _call_lstm_gradient_analysis(self, intent: Dict) -> Dict:
        """
        Call gradient analyzer for spatial comparisons
        
        Args:
            intent: Intent with metric
            
        Returns:
            Gradient analysis results
        """
        if self.gradient_analyzer is None:
            return {
                'type': 'gradient',
                'metric': intent['metric'],
                'error': 'Gradient analyzer not available'
            }
        
        try:
            metric = intent['metric']
            
            # Compute spatial gradients
            gradients_df = self.gradient_analyzer.compute_spatial_gradients(
                metric=metric,
                time_window='latest'
            )
            
            if len(gradients_df) == 0:
                return {
                    'type': 'gradient',
                    'metric': metric,
                    'error': 'No gradient data available (need multiple sensors)'
                }
            
            # Get top gradients
            top_gradients = gradients_df.nlargest(5, 'gradient')
            
            results = []
            for _, row in top_gradients.iterrows():
                results.append({
                    'sensor_pair': f"{row['sensor_a']} - {row['sensor_b']}",
                    'difference': row['difference'],
                    'distance': row['distance_m'],
                    'gradient': row['gradient']
                })
            
            return {
                'type': 'gradient',
                'metric': metric,
                'top_gradients': results
            }
            
        except Exception as e:
            return {
                'type': 'gradient',
                'metric': intent['metric'],
                'error': str(e)
            }
    
    def _explain_results(self, query: str, intent: Dict, lstm_output: Dict) -> str:
        """
        Generate human-readable explanation of LSTM results using the LLM.
        
        Args:
            query: Original user query
            intent: Parsed intent
            lstm_output: LSTM output data
            
        Returns:
            Human-readable explanation, or an error message if the LLM is
            unavailable.
        """
        # LLM is required – no template fallback
        if self.llm_model is None:
            return (
                "❌ LLM model is required for result explanation. "
                "Provide --mtnsails-model with a valid model path, "
                "or install dependencies: pip install optimum[onnxruntime] transformers"
            )
        return self._llm_explain_results(query, lstm_output)
    
    def _log_to_buffer(self, user_query: str, intent: Dict, lstm_output: Dict) -> None:
        """
        Append a validated interaction to the JSONL retrain buffer.

        Builds the training text in the same conversation format used by
        ``TaberEnviroTrainer`` so the daemon can fine-tune directly from
        the buffer records.

        Args:
            user_query: Original user query.
            intent: Parsed intent (type, metric, duration, sensor_id).
            lstm_output: Successful LSTM result dict (no 'error' key).
        """
        try:
            from llm_interface.retrain_buffer import append_validated_interaction
            from llm_interface.taber_enviro_trainer import _INTENT_TEMPLATE

            # Build training text matching the bridge intent format
            intent_text = _INTENT_TEMPLATE.format(
                query_type=intent["type"],
                metric=intent["metric"],
                duration=intent.get("duration", 60),
            )
            training_text = f"User: {user_query}\nAssistant: {intent_text}"

            append_validated_interaction(
                buffer_dir=self.buffer_dir,
                user_query=user_query,
                parsed_intent=intent,
                lstm_output=lstm_output,
                training_text=training_text,
                llm_intent_raw_response=self._last_llm_raw_response,
                intent_prompt=self._last_intent_prompt,
            )
            print(f"✓ Interaction logged to retrain buffer: {self.buffer_dir}")
        except Exception as exc:
            # Buffer logging failure must never break the main response flow
            print(f"WARNING: Failed to log to retrain buffer: {exc}")
    
    def _llm_explain_results(self, query: str, lstm_output: Dict) -> str:
        """
        Use LLM to generate human explanation of LSTM results
        
        Args:
            query: Original query
            lstm_output: LSTM results
            
        Returns:
            Natural language explanation
        """
        prompt = f"""User asked: {query}

LSTM Results:
{json.dumps(lstm_output, indent=2)}

Explain these results in clear, human-friendly language. Be concise but informative.
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the explanation part (after the prompt)
        if "Explain these results" in response:
            response = response.split("Explain these results")[-1].strip()
        
        return response
    
    def _template_explain_results(self, intent: Dict, lstm_output: Dict) -> str:
        """
        Template-based explanation as fallback
        
        Args:
            intent: Parsed intent
            lstm_output: LSTM results
            
        Returns:
            Formatted explanation
        """
        if 'error' in lstm_output:
            return f"❌ {lstm_output['error']}"
        
        output_type = lstm_output.get('type', intent['type'])
        
        if output_type == 'forecast':
            return self._explain_forecast(lstm_output)
        elif output_type == 'anomaly':
            return self._explain_anomaly(lstm_output)
        elif output_type == 'gradient':
            return self._explain_gradient(lstm_output)
        else:
            return json.dumps(lstm_output, indent=2)
    
    def _explain_forecast(self, output: Dict) -> str:
        """Format forecast explanation"""
        metric = output['metric']
        metric_name = {
            'temp': 'Temperature',
            'humidity': 'Humidity',
            'light': 'Light',
            'barometer': 'Barometric Pressure'
        }.get(metric, metric)
        
        unit = {
            'temp': '°C',
            'humidity': '%',
            'light': 'lux',
            'barometer': 'hPa'
        }.get(metric, '')
        
        current = output.get('current_value')
        predictions = output.get('predictions', [])
        steps = output.get('steps_ahead', len(predictions))
        
        lines = [f"📊 {metric_name} Forecast\n"]
        
        if current is not None:
            lines.append(f"Current: {current:.2f}{unit}")
        
        if len(predictions) > 0:
            # Show predictions at key intervals
            intervals = [
                (min(15, len(predictions)-1), "15 min"),
                (min(30, len(predictions)-1), "30 min"),
                (min(60, len(predictions)-1), "1 hour"),
                (len(predictions)-1, f"{steps} min")
            ]
            
            seen_indices = set()
            for idx, label in intervals:
                if idx not in seen_indices and idx < len(predictions):
                    val = predictions[idx]
                    change = ""
                    if current is not None:
                        diff = val - current
                        change = f" ({diff:+.2f}{unit})"
                    lines.append(f"In {label}: {val:.2f}{unit}{change}")
                    seen_indices.add(idx)
        
        # Add trend analysis
        if len(predictions) >= 2 and current is not None:
            final = predictions[-1]
            if final > current + 0.5:
                lines.append(f"\n📈 Trend: Rising")
            elif final < current - 0.5:
                lines.append(f"\n📉 Trend: Falling")
            else:
                lines.append(f"\n➡️ Trend: Stable")
        
        return "\n".join(lines)
    
    def _explain_anomaly(self, output: Dict) -> str:
        """Format anomaly explanation"""
        metric = output['metric']
        severity = output.get('severity', 'Unknown')
        score = output.get('anomaly_score', 0)
        error = output.get('error', 0)
        
        icon = '✓' if severity == 'Low' else '⚠️' if severity == 'Medium' else '🚨'
        
        lines = [
            f"{icon} Anomaly Detection: {severity.upper()} SEVERITY\n",
            f"Metric: {metric}",
            f"Anomaly Score: {score:.2f}/1.0",
            f"Prediction Error: {error:.2f}"
        ]
        
        if severity == 'High':
            lines.append("\n⚠️ ALERT: Significant deviation detected!")
            lines.append("Recommended actions:")
            lines.append("• Verify sensor hardware")
            lines.append("• Check for environmental changes")
            lines.append("• Review recent system modifications")
        elif severity == 'Medium':
            lines.append("\n⚠️ WARNING: Moderate deviation detected")
            lines.append("Monitor closely for further changes")
        else:
            lines.append("\n✓ No significant anomalies detected")
            lines.append("Sensor operating within normal parameters")
        
        return "\n".join(lines)
    
    def _explain_gradient(self, output: Dict) -> str:
        """Format gradient explanation"""
        metric = output['metric']
        gradients = output.get('top_gradients', [])
        
        metric_name = {
            'temp': 'Temperature',
            'humidity': 'Humidity', 
            'light': 'Light',
            'barometer': 'Pressure'
        }.get(metric, metric)
        
        lines = [f"📍 {metric_name} Spatial Gradients\n"]
        
        if len(gradients) == 0:
            lines.append("No gradient data available (multiple sensors required)")
        else:
            lines.append("Top sensor pairs with largest differences:\n")
            for i, g in enumerate(gradients[:5], 1):
                lines.append(
                    f"{i}. {g['sensor_pair']}: "
                    f"{g['difference']:.2f} over {g['distance']:.1f}m "
                    f"(gradient: {g['gradient']:.3f}/m)"
                )
            
            # Add insight
            max_gradient = gradients[0]['gradient']
            if max_gradient > 0.5:
                lines.append(f"\n💡 Strong gradients detected - potential for energy harvesting")
            elif max_gradient > 0.2:
                lines.append(f"\n💡 Moderate gradients - consider airflow optimization")
            else:
                lines.append(f"\n💡 Uniform distribution - environment is well balanced")
        
        return "\n".join(lines)


def interactive_mode(bridge: MTNSailsLSTMBridge):
    """Run interactive query session"""
    print("\n" + "="*60)
    print("MTN Sails + Taber LSTM - Natural Language Interface")
    print("="*60)
    print("\nExample queries:")
    print("  • What will temperature be in 2 hours?")
    print("  • Are there any unusual readings?")
    print("  • Which sensors have the biggest temperature difference?")
    print("\nType 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            print()
            response = bridge.ask(query)
            print(f"🤖 Assistant:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Natural language interface for LSTM sensor predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python mtnsails_bridge.py --taber-model outputs/ --interactive
  
  # With MTN Sails LLM
  python mtnsails_bridge.py --mtnsails-model onnx_model --taber-model outputs/ --interactive
  
  # Single query
  python mtnsails_bridge.py --taber-model outputs/ --query "What will temp be in 2 hours?"
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
        required=True,
        help='Path to Taber LSTM model directory (required)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to process'
    )
    
    parser.add_argument(
        '--buffer-dir',
        type=str,
        default=None,
        help='Path to JSONL retrain buffer directory (optional; logs validated interactions)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Must specify either --interactive or --query")
    
    # Initialize bridge
    print("Initializing MTN Sails <-> LSTM Bridge...\n")
    bridge = MTNSailsLSTMBridge(
        mtnsails_model_path=args.mtnsails_model,
        taber_model_path=args.taber_model,
        buffer_dir=args.buffer_dir,
    )
    print("\n✓ Bridge initialized successfully\n")
    
    # Run requested mode
    if args.interactive:
        interactive_mode(bridge)
    elif args.query:
        response = bridge.ask(args.query)
        print(f"\n{response}\n")


if __name__ == '__main__':
    main()

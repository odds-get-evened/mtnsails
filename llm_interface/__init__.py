"""
LLM Natural Language Interface for LSTM Predictions

This package provides a natural language interface to the Taber LSTM pipeline
using MTN Sails LLM for query parsing and result explanation.

Architecture:
    User Query (Natural Language)
        ↓
    MTN Sails LLM (Parse Intent)
        ↓
    Taber LSTM Predictor (Numerical Predictions)
        ↓
    MTN Sails LLM (Generate Explanation)
        ↓
    Human-Readable Response
"""

from .mtnsails_bridge import MTNSailsLSTMBridge
from .training_data_generator import create_iot_analyst_training_data
from .taber_enviro_trainer import TaberEnviroTrainer

__all__ = ['MTNSailsLSTMBridge', 'create_iot_analyst_training_data', 'TaberEnviroTrainer']
__version__ = '1.0.0'

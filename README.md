# Federated Learning based Heart Disease Prediction

A comprehensive federated learning system for heart disease prediction that implements advanced security mechanisms to defend against adversarial attacks in collaborative machine learning environments.

## Overview

This is a research-oriented federated learning framework that enables multiple clients to collaboratively train a machine learning model for heart disease prediction without sharing their raw data. The system includes sophisticated defense mechanisms against malicious clients and provides comprehensive testing capabilities for evaluating federated learning security.

## Features

### Core Federated Learning
- **Distributed Training**: Multi-client federated learning implementation using the Flower framework 
- **Heart Disease Prediction**: Machine learning pipeline for cardiovascular disease classification 
- **Data Preprocessing**: Comprehensive data cleaning, feature engineering, and scaling pipeline 

### Security and Defense Mechanisms
- **Gradient Signature Verification (GSV)**: Advanced defense system against adversarial attacks 
- **Malicious Client Detection**: Real-time identification and blocking of compromised clients 
- **Attack Simulation**: Configurable malicious client behavior for security testing 

### Data Augmentation and Client Management
- **Client Data Augmentation**: Enhanced data diversity for improved model performance 
- **Stratified Data Distribution**: Balanced data partitioning across federated clients 

### Testing and Validation
- **Comprehensive Test Suite**: Automated testing framework for federated learning scenarios 
- **Performance Metrics**: Detailed evaluation including precision, recall, F1-score, and accuracy 

## Installation

### Prerequisites
- Python 3.8+
- Required packages: scikit-learn, pandas, numpy, flwr (Flower), optuna

### Setup
```bash
git clone https://github.com/Iruos8805/Federated-learning-based-Heart-Disease-Prediction.git
cd Federated-learning-based-Heart-Disease-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Standard Machine Learning Pipeline
Run the basic heart disease prediction model with hyperparameter optimization: 

```bash
# Basic training
python main.py

# Hyperparameter optimization
python main.py optuna
```

### Federated Learning

#### 1. Start the Federated Learning Server 

```bash
python fl_server.py
```

#### 2. Start Federated Learning Clients 

```bash
# Normal client
python fl_client.py 1

# Malicious client with continuous attacks
python fl_client.py 2 --malicious

# Malicious client with random attacks
python fl_client.py 3 --malicious --random --attack_probability 0.3
```

### Secure Federated Learning with GSV Defense 

```bash
# Start server with GSV defense
python fl_server_mit.py --num_clients 5 --num_rounds 30 --warmup 15

# Connect clients (same as above)
python fl_client.py 1
python fl_client.py 2 --malicious
```

## Project Structure

### Core Components
- **Dataset Management**: Heart disease dataset loading and basic preprocessing 
- **Federated Learning Server**: Coordinates training across distributed clients 
- **Federated Learning Client**: Individual client implementation with malicious behavior simulation 

### Security Implementation
- **GSV Strategy**: Gradient-based anomaly detection for identifying malicious updates 
- **Malicious Client Simulation**: Dedicated implementation for testing defense mechanisms 

### Testing Framework
- **Automated Testing**: Comprehensive validation of federated learning scenarios and security mechanisms 

## Configuration Options

### Client Configuration
- **Client ID**: Unique identifier for each federated client
- **Malicious Behavior**: Enable adversarial attacks for testing
- **Attack Patterns**: Continuous or random attack strategies
- **Warmup Rounds**: Normal behavior period before attacks begin
- **Attack Probability**: Frequency of attacks in random mode

### Server Configuration
- **Number of Clients**: Minimum required participants
- **Training Rounds**: Total federated learning iterations
- **Warmup Period**: Initial rounds before defense activation
- **GSV Parameters**: Threshold and detection sensitivity settings

## Security Features

### Gradient Signature Verification (GSV)
The system implements a novel defense mechanism that analyzes gradient patterns to detect malicious clients:
- **Multi-metric Analysis**: Combines gradient magnitude and sign pattern analysis
- **Adaptive Thresholds**: Dynamic adjustment based on client behavior
- **Client Profiling**: Individual behavioral fingerprinting
- **Persistent Blocking**: Permanent exclusion of detected malicious clients

## Logging and Monitoring

All federated learning activities are automatically logged:
- **Server Logs**: Comprehensive training progress and defense statistics
- **Client Logs**: Individual client training and attack behavior
- **GSV Metrics**: Detailed security analysis and detection reports
- **Test Reports**: Automated validation results with performance metrics

## Contributing

This project is designed for research purposes in federated learning security. Contributions should focus on:
- Enhanced defense mechanisms
- Additional attack strategies
- Improved evaluation metrics
- Extended test scenarios

## Notes

This implementation provides a complete federated learning ecosystem with integrated security mechanisms specifically designed for healthcare applications. The system demonstrates effective defense against adversarial attacks while maintaining high model performance in collaborative learning scenarios. The comprehensive testing framework ensures reliability and provides benchmarks for federated learning security research.

"""
Federated Learning Server with Biological Anomaly Filtering (BAF)
Author: [Your Name]
Description: Main server implementation with immune system defense against adversarial attacks
"""

import flwr as fl
import os
import sys
from datetime import datetime
from mitigation import BAFStrategy, load_validation_data  # NOTE: load_validation_data should not augment

#---------------------------------------------------------------
# Logging setup
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"logs/fl_server_{timestamp}.txt", "w")

class Tee:
    """Helper class to redirect output to both console and log file"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

# Redirect stdout and stderr
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
#---------------------------------------------------------------


def weighted_average(metrics):
    """
    Aggregate evaluation metrics from all clients.
    """
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_recall = sum(recalls) / sum(examples)

    print(f"=== GLOBAL MODEL PERFORMANCE ===")
    print(f"Aggregated Recall Score: {aggregated_recall:.4f}")
    print(f"Total Examples: {sum(examples)}")
    print(f"Individual Client Recalls: {[m['recall'] for _, m in metrics]}")
    print("=" * 35)

    return {"recall": aggregated_recall}


def start_server():
    """
    Start the federated learning server with BAF defense system
    """
    print("-" * 60)
    print("🚀 Starting Federated Learning Server with BAF Defense")
    print("-" * 60)

    # Load clean validation data (without augmentation)
    print("🔧 Initializing Biological Anomaly Filter...")
    try:
        validation_data = load_validation_data()  # This should now return raw or minimally preprocessed data
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        raise

    # Create strategy
    try:
        strategy = BAFStrategy(
            validation_data=validation_data,
            min_fit_clients=3,
            min_available_clients=3,
            min_evaluate_clients=3,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    except Exception as e:
        print(f"❌ Error creating BAF strategy: {e}")
        raise

    print("🦠 Biological Anomaly Filtering (BAF) activated")
    print("🔬 Immune system ready for adversarial defense")
    print("🧬 Features:")
    print("   - Functional fingerprinting for behavioral analysis")
    print("   - Adaptive immune memory with self-profile evolution")
    print("   - Multi-layered anomaly detection")
    print("   - Affinity maturation for dynamic sensitivity")
    print("-" * 60)

    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=20),
            strategy=strategy
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise

    print("\n" + "=" * 60)
    print("🔬 Final Immune System Status:")
    try:
        immune_status = strategy.get_immune_status()
        for key, value in immune_status.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Error getting immune status: {e}")
    print("=" * 60)


if __name__ == '__main__':
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'log_file' in locals():
            log_file.close()
        print("📝 Logs saved to logs/ directory")

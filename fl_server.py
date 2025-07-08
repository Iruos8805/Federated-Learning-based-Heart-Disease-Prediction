import flwr as fl
import os
import sys
from datetime import datetime

#---------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"logs/fl_server_{timestamp}.txt", "w")

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)
#---------------------------------------------------------------

def weighted_average(metrics):
    """Aggregate evaluation metrics from all clients."""
    # Calculate weighted average of recall scores
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Weighted average
    aggregated_recall = sum(recalls) / sum(examples)
    
    print(f"=== GLOBAL MODEL PERFORMANCE ===")
    print(f"Aggregated Recall Score: {aggregated_recall:.4f}")
    print(f"Total Examples: {sum(examples)}")
    print(f"Individual Client Recalls: {[m['recall'] for _, m in metrics]}")
    print("=" * 35)
    
    return {"recall": aggregated_recall}

def start_server():
    print("-" * 60)
    print("Starting Federated Learning Server...")
    print("-" * 60)

    # Custom strategy with better configuration
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3,
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Use all available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate client metrics
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),  # Increased rounds for better convergence
        strategy=strategy
    )

if __name__ == '__main__':
    start_server()
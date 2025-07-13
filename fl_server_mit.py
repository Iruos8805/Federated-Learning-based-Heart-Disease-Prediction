"""
Federated Learning Server with Gradient Signature Verification (GSV)
Description: Main server implementation with GSV defense against adversarial attacks
"""

import flwr as fl
import os
import sys
import atexit
import argparse
from datetime import datetime
from mitigation import GSVStrategy, load_validation_data

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Server with GSV Defense')
    parser.add_argument('--num_clients', '-n', type=int, default=3, 
                       help='Number of clients to use (default: 3)')
    parser.add_argument('--num_rounds', '-r', type=int, default=30,
                       help='Number of training rounds (default: 30)')
    parser.add_argument('--warmup', type=int, default=15,
                       help='Number of warmup rounds (default: 15)')
    return parser.parse_args()

args = parse_args()
num_clients = args.num_clients
num_rounds = args.num_rounds
warmup_rounds = args.warmup

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = open(f"logs/fl_server_gsv_{num_clients}clients_{timestamp}.txt", "w")

original_stdout = sys.stdout
original_stderr = sys.stderr

class Tee:
    """Helper class to redirect output to both console and log file"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                if hasattr(s, 'write') and not s.closed:
                    s.write(data)
                    s.flush()
            except (ValueError, AttributeError):
                pass

    def flush(self):
        for s in self.streams:
            try:
                if hasattr(s, 'flush') and not s.closed:
                    s.flush()
            except (ValueError, AttributeError):
                pass

def cleanup_logging():
    """Cleanup function to restore original streams and close log file"""
    try:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file and not log_file.closed:
            log_file.close()
    except:
        pass

atexit.register(cleanup_logging)

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


def weighted_average(metrics):
    """
    Aggregate evaluation metrics from all clients.
    """
    if not metrics:
        print("No metrics to aggregate")
        return {"recall": 0.0}
        
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_recall = sum(recalls) / sum(examples)

    print(f"=== GLOBAL MODEL PERFORMANCE ===")
    print(f"Aggregated Recall Score: {aggregated_recall:.4f}")
    print(f"Total Examples: {sum(examples)}")
    print(f"Participating Clients: {len(metrics)}")
    print(f"Individual Client Recalls: {[m['recall'] for _, m in metrics]}")
    print("=" * 35)

    return {"recall": aggregated_recall}


def start_server():
    """
    Start the federated learning server with GSV defense system
    """
    print("-" * 60)
    print("Starting Federated Learning Server with GSV Defense")
    print(f"Configuration: {num_clients} clients, {num_rounds} rounds, {warmup_rounds} warmup")
    print("-" * 60)

    print("Initializing Gradient Signature Verifier...")
    try:
        validation_data = load_validation_data()
        print("Validation data loaded successfully")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        print("Proceeding without validation data (GSV doesn't require it)")
        validation_data = None

    try:
        strategy = GSVStrategy(
            warmup_rounds=warmup_rounds,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            min_evaluate_clients=num_clients,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        print("GSV strategy initialized successfully")
        print(f"   • Minimum clients required: {num_clients}")
        print(f"   • Training rounds: {num_rounds}")
        print(f"   • Warmup rounds: {warmup_rounds}")
    except Exception as e:
        print(f"Error creating GSV strategy: {e}")
        raise

    print("Gradient Signature Verification (GSV) activated")
    print("Defense system ready for adversarial attack detection")
    print("GSV Features:")
    print("   - Gradient magnitude distribution analysis")
    print("   - Sign pattern behavioral fingerprinting")
    print("   - Client-specific profile evolution")
    print("   - Adaptive threshold adjustment")
    print("   - Multi-metric anomaly detection")
    print("   - Individual client filtering (not all clients)")
    print("   - Persistent client blocking after detection")
    print("-" * 60)

    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        raise
    finally:
        cleanup_logging()

    print("\n" + "=" * 60)
    print("Final GSV System Status:")
    try:
        gsv_status = strategy.get_verification_status()
        for key, value in gsv_status.items():
            print(f"   {key}: {value}")
        
        print("\nDetailed Defense Report:")
        print(f"   • Total rounds completed: {gsv_status['round']}")
        print(f"   • Malicious clients blocked: {gsv_status['blocked_clients']}")
        print(f"   • Attack detection rate: {gsv_status['attacks_detected']}/{gsv_status['total_processed']}")
        print(f"   • System filter rate: {gsv_status['filter_rate']:.2%}")
        if gsv_status['blocked_client_ids']:
            print(f"   • Blocked client IDs: {gsv_status['blocked_client_ids']}")
            
    except Exception as e:
        print(f"Error getting GSV status: {e}")
    print("=" * 60)


if __name__ == '__main__':
    print(f"Server starting with {num_clients} clients for {num_rounds} rounds ({warmup_rounds} warmup)")
    print("Usage: python fl_server_mit.py --num_clients 5 --num_rounds 25 --warmup 10")
    print("   or: python fl_server_mit.py -n 5 -r 25 --warmup 10")
    print("-" * 60)
    
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nServer shutdown requested by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Logs saved to logs/ directory")
        cleanup_logging()
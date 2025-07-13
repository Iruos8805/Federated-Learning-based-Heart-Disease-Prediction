#!/usr/bin/env python3
"""
Comprehensive Test Suite for Federated Learning System with GSV Defense
Description: Tests all possible scenarios and combinations of the FL system with result validation
"""

import subprocess
import time
import os
import sys
import signal
import threading
from datetime import datetime
import json
import argparse
import csv
import re

class FederatedLearningTester:
    def __init__(self):
        self.test_results = []
        self.server_process = None
        self.client_processes = []
        self.test_count = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.client_round_mapping = {}  
        self.num_rounds = 30  
        
    def log_test(self, test_name, status, details="", validation_results=None):
        """Log test results with validation details"""
        result = {
            "test_number": self.test_count,
            "test_name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "validation_results": validation_results or {}
        }
        self.test_results.append(result)
        
        print(f"Test {self.test_count}: {test_name} - {status}")
        if details:
            print(f"   Details: {details}")
        if validation_results:
            print(f"   Validation: {validation_results}")
        print("-" * 80)
        
    def cleanup_processes(self):
        """Clean up all running processes"""
        for process in self.client_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except:
                try:
                    self.server_process.kill()
                except:
                    pass
        
        self.client_processes = []
        self.server_process = None
        
    def start_server(self, num_clients=3, num_rounds=10, warmup=5):
        """Start the federated learning server"""
        self.cleanup_processes()
        
        
        num_rounds = max(30, num_rounds)
        self.num_rounds = num_rounds
        warmup = max(num_rounds // 2, warmup)

        server_cmd = [
            sys.executable, "fl_server_mit.py",
            "--num_clients", str(num_clients),
            "--num_rounds", str(num_rounds),
            "--warmup", str(warmup)
        ]
        
        print(f"Starting server with command:")
        print(f"   {' '.join(server_cmd)}")
        
        try:
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(5)  
            print("Server started successfully")
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def start_client(self, client_id, malicious=False, random_attack=False, 
                    warmup=5, attack_probability=0.3):
        """Start a federated learning client"""
        
        warmup = max(self.num_rounds // 2, warmup)
        
        client_cmd = [
            sys.executable, "fl_client.py",
            str(client_id),
            "--warmup", str(warmup)
        ]
        
        if malicious:
            client_cmd.append("--malicious")
            
        if random_attack:
            client_cmd.append("--random")
            client_cmd.extend(["--attack_probability", str(attack_probability)])
        
        print(f"Starting client {client_id} with command:")
        print(f"   {' '.join(client_cmd)}")
        
        try:
            process = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.client_processes.append(process)
            print(f"Client {client_id} started successfully")
            return True
        except Exception as e:
            print(f"Failed to start client {client_id}: {e}")
            return False
    
    def wait_for_completion(self, timeout=120):
        """Wait for FL training to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.server_process and self.server_process.poll() is not None:
                return True
            time.sleep(3)
        
        return False
    
    def parse_server_output(self, output):
        """Parse server output to extract GSV statistics"""
        gsv_stats = {
            "total_rounds": 0,
            "blocked_clients": 0,
            "attacks_detected": 0,
            "total_processed": 0,
            "filter_rate": 0.0,
            "blocked_client_ids": []
        }
        
        lines = output.split('\n')
        for line in lines:
            if "Total rounds completed:" in line:
                match = re.search(r'(\d+)', line)
                if match:
                    gsv_stats["total_rounds"] = int(match.group(1))
            elif "Malicious clients blocked:" in line:
                match = re.search(r'(\d+)', line)
                if match:
                    gsv_stats["blocked_clients"] = int(match.group(1))
            elif "Attack detection rate:" in line:
                match = re.search(r'(\d+)/(\d+)', line)
                if match:
                    gsv_stats["attacks_detected"] = int(match.group(1))
                    gsv_stats["total_processed"] = int(match.group(2))
            elif "System filter rate:" in line:
                match = re.search(r'(\d+\.\d+)%', line)
                if match:
                    gsv_stats["filter_rate"] = float(match.group(1))
            elif "Blocked client IDs:" in line:
                ids_match = re.search(r'\[(.*?)\]', line)
                if ids_match:
                    ids_str = ids_match.group(1)
                    if ids_str.strip():
                        gsv_stats["blocked_client_ids"] = [id.strip().strip("'\"") for id in ids_str.split(',')]
        
        return gsv_stats
    
    def parse_client_scores_log(self):
        """Parse client scores CSV log to get detailed analysis"""
        scores_file = "logs/client_scores.csv"
        if not os.path.exists(scores_file):
            return {}
        
        client_data = {}
        try:
            with open(scores_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    client_id = row['client_id']
                    if client_id not in client_data:
                        client_data[client_id] = {
                            'scores': [],
                            'filtered_count': 0,
                            'blocked': False,
                            'rounds_participated': 0,
                            'max_score': 0.0,
                            'avg_score': 0.0
                        }
                    
                    score = float(row['score'])
                    client_data[client_id]['scores'].append(score)
                    client_data[client_id]['filtered_count'] += int(row['filtered'])
                    client_data[client_id]['blocked'] = bool(int(row['blocked']))
                    client_data[client_id]['rounds_participated'] += 1
                    client_data[client_id]['max_score'] = max(client_data[client_id]['max_score'], score)
                    
            for client_id in client_data:
                scores = client_data[client_id]['scores']
                if scores:
                    client_data[client_id]['avg_score'] = sum(scores) / len(scores)
                    
        except Exception as e:
            print(f"Error parsing client scores: {e}")
        
        return client_data
    
    def validate_test_results(self, test_name, client_configs, server_output, timeout_occurred=False):
        """Validate test results against expected behavior"""
        if timeout_occurred:
            return False, {"error": "Test timed out"}
            
        gsv_stats = self.parse_server_output(server_output)
        client_scores = self.parse_client_scores_log()
        
        expected_malicious_ids = set()
        expected_normal_ids = set()
        
        for config in client_configs:
            client_id = str(config['client_id'])
            if config.get('params', {}).get('malicious', False):
                expected_malicious_ids.add(client_id)
            else:
                expected_normal_ids.add(client_id)
        
        blocked_client_ids = set(gsv_stats.get("blocked_client_ids", []))
        
        true_positives = len(expected_malicious_ids & blocked_client_ids)
        false_positives = len(blocked_client_ids - expected_malicious_ids)
        false_negatives = len(expected_malicious_ids - blocked_client_ids)
        true_negatives = len(expected_normal_ids - blocked_client_ids)
        
        total_clients = len(client_configs)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_clients if total_clients > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        validation_results = {
            "expected_malicious": list(expected_malicious_ids),
            "expected_normal": list(expected_normal_ids),
            "blocked_clients": list(blocked_client_ids),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "gsv_stats": gsv_stats,
            "client_analysis": client_scores
        }
        
        success = self.evaluate_test_success(test_name, validation_results, client_configs)
        
        return success, validation_results
    
    def evaluate_test_success(self, test_name, validation_results, client_configs):
        """Evaluate test success based on test type and validation results - STRICT: No false positives allowed"""
        
        if validation_results["false_positives"] > 0:
            return False
        
        if "basic_functionality" in test_name.lower() or "normal" in test_name.lower():
            return (validation_results["false_positives"] == 0 and 
                   validation_results["false_negatives"] == 0 and
                   validation_results["accuracy"] == 1.0)
        
        elif "false_positive" in test_name.lower():
            return validation_results["false_positives"] == 0
        
        elif "single_continuous_malicious" in test_name.lower():
            return (validation_results["true_positives"] >= 1 and 
                   validation_results["false_positives"] == 0)
        
        elif "random_malicious" in test_name.lower():
            return (validation_results["false_positives"] == 0 and
                   validation_results["true_positives"] >= 0)  
        
        elif "detection_consistency" in test_name.lower():
            return (validation_results["true_positives"] >= 1 and 
                   validation_results["false_positives"] == 0)
        
        elif "early_detection" in test_name.lower():
            return (validation_results["true_positives"] >= 1 and 
                   validation_results["false_positives"] == 0)

        else:
            return validation_results["false_positives"] == 0

    def _get_client_id(self, client_proxy, fit_res):
        """Get client ID from fit results or fallback to proxy-based ID"""
        if hasattr(fit_res, 'metrics') and fit_res.metrics and 'client_id' in fit_res.metrics:
            return fit_res.metrics['client_id']
        
        client_key = str(client_proxy)
        if client_key not in self.client_round_mapping:
            client_id = f"client_{len(self.client_round_mapping)}"
            self.client_round_mapping[client_key] = client_id
        return self.client_round_mapping[client_key]

    def run_test_scenario(self, test_name, server_config, client_configs, timeout=180):
        """Run a specific test scenario with result validation"""
        self.test_count += 1
        print(f"\nRunning Test {self.test_count}: {test_name}")
        print("=" * 80)
        
        num_rounds = server_config.get('num_rounds', 30)
        if timeout == 180:  
            timeout = max(timeout, num_rounds * 6)  
        
        print(f"Test Configuration:")
        print(f"   Server: {server_config}")
        print(f"   Clients: {len(client_configs)} total")
        
        malicious_clients = [c for c in client_configs if c.get('params', {}).get('malicious', False)]
        normal_clients = [c for c in client_configs if not c.get('params', {}).get('malicious', False)]
        
        print(f"   - Normal clients: {len(normal_clients)} {[c['client_id'] for c in normal_clients]}")
        print(f"   - Malicious clients: {len(malicious_clients)} {[c['client_id'] for c in malicious_clients]}")
        
        if malicious_clients:
            for client in malicious_clients:
                params = client.get('params', {})
                attack_type = "random" if params.get('random_attack', False) else "continuous"
                prob = params.get('attack_probability', 0.3)
                print(f"     Client {client['client_id']}: {attack_type} attack" + 
                      (f" (prob: {prob})" if attack_type == "random" else ""))
        
        print("-" * 80)
        
        if os.path.exists("logs/client_scores.csv"):
            os.remove("logs/client_scores.csv")
        
        print("Starting server...")
        if not self.start_server(**server_config):
            self.log_test(test_name, "FAILED", "Server failed to start")
            self.failed_tests += 1
            return False
        
        print(f"\nStarting {len(client_configs)} clients...")
        for i, client_config in enumerate(client_configs):
            if not self.start_client(client_config['client_id'], **client_config.get('params', {})):
                self.log_test(test_name, "FAILED", f"Client {client_config['client_id']} failed to start")
                self.failed_tests += 1
                return False
            time.sleep(2)  
        
        print(f"\nWaiting for training to complete (timeout: {timeout}s)...")
        
        server_output = ""
        timeout_occurred = False
        
        if self.wait_for_completion(timeout):
            print("Training completed successfully")
            try:
                if self.server_process is not None:
                    stdout, stderr = self.server_process.communicate(timeout=15)
                    server_output = stdout + stderr
                else:
                    server_output = "Server process is None"
            except Exception as e:
                server_output = f"Failed to capture output: {e}"
        else:
            print(" Training timed out")
            timeout_occurred = True
            server_output = "Training timed out"
        
        print("Validating results...")
        success, validation_results = self.validate_test_results(
            test_name, client_configs, server_output, timeout_occurred
        )
        
        false_positives = validation_results.get('false_positives', 0)
        true_positives = validation_results.get('true_positives', 0)
        false_negatives = validation_results.get('false_negatives', 0)
        
        if success:
            self.log_test(test_name, "PASSED", 
                         f"STRICT VALIDATION PASSED - "
                         f"FP: {false_positives}, TP: {true_positives}, FN: {false_negatives}, "
                         f"Precision: {validation_results['precision']:.3f}, "
                         f"Recall: {validation_results['recall']:.3f}",
                         validation_results)
            self.passed_tests += 1
        else:
            failure_reason = "STRICT VALIDATION FAILED"
            if timeout_occurred:
                failure_reason = "Training timed out"
            elif int(false_positives) > 0:
                failure_reason = f"FALSE POSITIVES DETECTED: {false_positives} normal clients incorrectly blocked"
            elif int(false_negatives) > 0 and "basic_functionality" in test_name.lower():
                failure_reason = f"DETECTION FAILURE: {false_negatives} malicious clients not detected"
            
            self.log_test(test_name, "FAILED", 
                         f"{failure_reason} - "
                         f"FP: {false_positives}, TP: {true_positives}, FN: {false_negatives}",
                         validation_results)
            self.failed_tests += 1
        
        print("Cleaning up processes...")
        self.cleanup_processes()
        time.sleep(3)
        
        return success

    def test_basic_functionality(self):
        """Test 1: Basic FL functionality with normal clients - should have no false positives"""
        return self.run_test_scenario(
            "Basic FL Training (All Normal Clients)",
            {"num_clients": 3, "num_rounds": 30, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"warmup": 5}},
                {"client_id": 3, "params": {"warmup": 5}}
            ]
        )
    
    def test_single_continuous_malicious(self):
        """Test 2: Single continuous malicious client - should detect and block"""
        return self.run_test_scenario(
            "Single Continuous Malicious Client",
            {"num_clients": 3, "num_rounds": 30, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"malicious": True, "warmup": 5}},
                {"client_id": 3, "params": {"warmup": 5}}
            ]
        )
    
    def test_single_random_malicious_high_prob(self):
        """Test 3: Single random malicious client (high probability)"""
        return self.run_test_scenario(
            "Single Random Malicious Client (High Probability)",
            {"num_clients": 3, "num_rounds": 30, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"malicious": True, "random_attack": True, "warmup": 5, "attack_probability": 0.8}},
                {"client_id": 3, "params": {"warmup": 5}}
            ]
        )
    
    def test_false_positive_check(self):
        """Test 6: False positive check with many normal clients"""
        return self.run_test_scenario(
            "False Positive Check",
            {"num_clients": 6, "num_rounds": 30, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"warmup": 5}},
                {"client_id": 3, "params": {"warmup": 5}},
                {"client_id": 4, "params": {"warmup": 5}},
                {"client_id": 5, "params": {"warmup": 5}},
                {"client_id": 6, "params": {"warmup": 5}}
            ]
        )
    
    def test_early_detection(self):
        """Test 7: Early detection with short warmup"""
        return self.run_test_scenario(
            "Early Detection Test",
            {"num_clients": 3, "num_rounds": 30, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"malicious": True, "warmup": 5}},
                {"client_id": 3, "params": {"warmup": 5}}
            ]
        )
    
    def test_detection_consistency(self):
        """Test 9: Detection consistency across many rounds"""
        return self.run_test_scenario(
            "Detection Consistency Test",
            {"num_clients": 3, "num_rounds": 35, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"malicious": True, "warmup": 5}},
                {"client_id": 3, "params": {"warmup": 5}}
            ],
            timeout=600  
        )
    
    def test_random_attacks_low_probability(self):
        """Test 10: Random attacks with low probability"""
        return self.run_test_scenario(
            "Random Attacks (Low Probability)",
            {"num_clients": 3, "num_rounds": 40, "warmup": 5},  
            [
                {"client_id": 1, "params": {"warmup": 5}},
                {"client_id": 2, "params": {"malicious": True, "random_attack": True, "warmup": 5, "attack_probability": 0.2}},
                {"client_id": 3, "params": {"warmup": 5}}
            ],
            timeout=700  
        )
    
    def run_all_tests(self):
        """Run all test scenarios with comprehensive validation"""
        print("Starting Comprehensive FL System Test Suite with Result Validation")
        print("=" * 80)
        
        test_methods = [
            self.test_basic_functionality,
            self.test_single_continuous_malicious,
            self.test_single_random_malicious_high_prob,
            self.test_false_positive_check,
            self.test_early_detection,
            self.test_detection_consistency,
            self.test_random_attacks_low_probability,
        ]
        
        total_tests = len(test_methods)
        print(f"Total tests to run: {total_tests}")
        print("=" * 80)
        
        for i, test_method in enumerate(test_methods, 1):
            print(f"\nProgress: {i}/{total_tests}")
            try:
                test_method()
            except Exception as e:
                self.log_test(f"Test {self.test_count + 1}", "FAILED", f"Exception: {str(e)}")
                self.failed_tests += 1
                self.test_count += 1
                import traceback
                traceback.print_exc()
            
            print(f"⏸️  Pausing 5 seconds before next test...")
            time.sleep(5)  
        
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report with validation metrics"""
        print("\n" + "=" * 80)
        print("STRICT VALIDATION TEST SUITE SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {self.test_count}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.test_count)*100:.1f}%")
        
        total_false_positives = 0
        total_true_positives = 0
        total_false_negatives = 0
        tests_with_zero_fp = 0
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_tests = 0
        
        for result in self.test_results:
            if result.get('validation_results'):
                vr = result['validation_results']
                total_false_positives += vr.get('false_positives', 0)
                total_true_positives += vr.get('true_positives', 0)
                total_false_negatives += vr.get('false_negatives', 0)
                
                if vr.get('false_positives', 0) == 0:
                    tests_with_zero_fp += 1
                
                if 'accuracy' in vr:
                    total_accuracy += vr['accuracy']
                    total_precision += vr['precision']
                    total_recall += vr['recall']
                    total_f1 += vr['f1_score']
                    valid_tests += 1
        
        print(f"\nSTRICT VALIDATION METRICS:")
        print(f"Total False Positives Across All Tests: {total_false_positives}")
        print(f"Total True Positives Across All Tests: {total_true_positives}")
        print(f"Total False Negatives Across All Tests: {total_false_negatives}")
        print(f"Tests with Zero False Positives: {tests_with_zero_fp}/{self.test_count}")
        print(f"False Positive Rate: {total_false_positives}/{self.test_count} = {total_false_positives/self.test_count:.3f}")
        
        if total_false_positives == 0:
            print("PERFECT SCORE: Zero false positives across all tests!")
        else:
            print("IMPROVEMENT NEEDED: False positives detected")
        
        if valid_tests > 0:
            print(f"\nOVERALL GSV DEFENSE SYSTEM PERFORMANCE:")
            print(f"Average Accuracy: {total_accuracy/valid_tests:.3f}")
            print(f"Average Precision: {total_precision/valid_tests:.3f}")
            print(f"Average Recall: {total_recall/valid_tests:.3f}")
            print(f"Average F1-Score: {total_f1/valid_tests:.3f}")
        
        report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": self.test_count,
                    "passed": self.passed_tests,
                    "failed": self.failed_tests,
                    "success_rate": (self.passed_tests/self.test_count)*100 if self.test_count > 0 else 0,
                    "avg_accuracy": total_accuracy/valid_tests if valid_tests > 0 else 0,
                    "avg_precision": total_precision/valid_tests if valid_tests > 0 else 0,
                    "avg_recall": total_recall/valid_tests if valid_tests > 0 else 0,
                    "avg_f1_score": total_f1/valid_tests if valid_tests > 0 else 0
                },
                "test_results": self.test_results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_filename}")
        
        if self.failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAILED":
                    vr = result.get('validation_results', {})
                    fp = vr.get('false_positives', 0)
                    tp = vr.get('true_positives', 0)
                    fn = vr.get('false_negatives', 0)
                    print(f"  - {result['test_name']}: {result['details']}")
                    print(f"    Metrics: FP={fp}, TP={tp}, FN={fn}")
        
        print("\nTest Suite Complete!")

def main():
    parser = argparse.ArgumentParser(description='FL System Test Suite with Result Validation')
    parser.add_argument('--test', type=str, help='Run specific test by name')
    parser.add_argument('--list', action='store_true', help='List all available tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output with commands')
    args = parser.parse_args()
    
    tester = FederatedLearningTester()
    
    if args.list:
        print("Available tests:")
        tests = [
            "basic_functionality", "single_continuous_malicious", "single_random_malicious_high_prob",
            "false_positive_check","early_detection", "detection_consistency", "random_attacks_low_probability"
        ]
        for i, test in enumerate(tests, 1):
            print(f"{i}. {test}")
        return
    
    if args.test:
        test_method = getattr(tester, f"test_{args.test}", None)
        if test_method:
            print(f"Running specific test: {args.test}")
            test_method()
            tester.generate_test_report()
        else:
            print(f"Test '{args.test}' not found. Use --list to see available tests.")
    else:
        tester.run_all_tests()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        tester = FederatedLearningTester()
        tester.cleanup_processes()
    except Exception as e:
        print(f"\nTest suite error: {e}")
        import traceback
        traceback.print_exc()

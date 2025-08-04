"""Enhanced load testing for quantum circuit execution and ML workflows with comprehensive metrics."""

import pytest
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from unittest.mock import patch, MagicMock
import numpy as np
from collections import defaultdict
import statistics
import psutil
import os

from quantum_mlops.testing import QuantumTestCase, performance_benchmark
from quantum_mlops.core import QuantumMLPipeline, QuantumDevice


class TestQuantumLoadTesting(QuantumTestCase):
    """Enhanced load testing for quantum systems with comprehensive scenarios."""
    
    def setUp(self):
        """Set up load testing environment."""
        super().setUp()
        
        # Load testing parameters
        self.load_test_duration = 60  # seconds
        self.max_concurrent_circuits = 100
        self.max_workers = min(20, (os.cpu_count() or 1) + 4)
        self.performance_thresholds = {
            'max_latency': 10.0,  # seconds
            'min_throughput': 1.0,  # operations/second
            'max_error_rate': 0.05,  # 5%
            'max_memory_mb': 500  # MB
        }
        
        # Metrics tracking
        self.load_metrics = {
            'execution_times': [],
            'memory_snapshots': [],
            'error_counts': defaultdict(int),
            'throughput_samples': [],
            'concurrent_load_levels': []
        }
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_concurrent_quantum_circuit_execution_comprehensive(self):
        """Test concurrent execution of quantum circuits with comprehensive load analysis."""
        circuit_scenarios = [
            {'circuits': 10, 'qubits': 3, 'depth': 2, 'name': 'light_load'},
            {'circuits': 50, 'qubits': 4, 'depth': 3, 'name': 'medium_load'},
            {'circuits': 100, 'qubits': 4, 'depth': 4, 'name': 'heavy_load'},
            {'circuits': 200, 'qubits': 3, 'depth': 2, 'name': 'burst_load'}
        ]
        
        for scenario in circuit_scenarios:
            with self.subTest(load_scenario=scenario['name']):
                num_circuits = scenario['circuits']
                n_qubits = scenario['qubits']
                depth = scenario['depth']
                
                # Create test model
                model = self.create_model(
                    n_qubits=n_qubits, 
                    circuit_type='variational', 
                    depth=depth
                )
                
                # Generate test data
                X = np.random.random((num_circuits, n_qubits))
                
                # Track performance metrics
                start_time = time.time()
                execution_times = []
                memory_usage = []
                errors = 0
                
                def execute_circuit(circuit_id):
                    """Execute single quantum circuit with monitoring."""
                    try:
                        circuit_start = time.time()
                        
                        # Monitor memory before execution
                        try:
                            process = psutil.Process(os.getpid())
                            memory_before = process.memory_info().rss
                        except:
                            memory_before = 0
                        
                        # Execute circuit
                        prediction = model.predict(X[circuit_id:circuit_id+1])
                        
                        circuit_time = time.time() - circuit_start
                        
                        # Monitor memory after execution
                        try:
                            memory_after = process.memory_info().rss
                            memory_delta = memory_after - memory_before
                        except:
                            memory_delta = 0
                        
                        return {
                            'circuit_id': circuit_id,
                            'execution_time': circuit_time,
                            'memory_delta': memory_delta,
                            'prediction': prediction[0],
                            'success': True
                        }
                        
                    except Exception as e:
                        return {
                            'circuit_id': circuit_id,
                            'execution_time': time.time() - circuit_start,
                            'memory_delta': 0,
                            'error': str(e),
                            'success': False
                        }
                
                # Execute circuits concurrently
                max_workers = min(self.max_workers, num_circuits)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(execute_circuit, i) 
                        for i in range(num_circuits)
                    ]
                    
                    results = []
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            execution_times.append(result['execution_time'])
                            memory_usage.append(result['memory_delta'])
                        else:
                            errors += 1
                
                total_time = time.time() - start_time
                
                # Analyze performance metrics
                successful_circuits = len([r for r in results if r['success']])
                error_rate = errors / num_circuits
                throughput = successful_circuits / total_time
                avg_latency = np.mean(execution_times) if execution_times else float('inf')
                p95_latency = np.percentile(execution_times, 95) if execution_times else float('inf')
                avg_memory_mb = np.mean(memory_usage) / 1024 / 1024 if memory_usage else 0
                
                # Performance assertions based on load level
                if scenario['name'] == 'light_load':
                    self.assertLess(avg_latency, 1.0, "Light load should have low latency")
                    self.assertGreater(throughput, 5.0, "Light load should have high throughput")
                    self.assertLess(error_rate, 0.01, "Light load should have minimal errors")
                    
                elif scenario['name'] == 'medium_load':
                    self.assertLess(avg_latency, 3.0, "Medium load latency too high")
                    self.assertGreater(throughput, 2.0, "Medium load throughput too low")
                    self.assertLess(error_rate, 0.03, "Medium load error rate too high")
                    
                elif scenario['name'] == 'heavy_load':
                    self.assertLess(avg_latency, 5.0, "Heavy load latency excessive")
                    self.assertGreater(throughput, 1.0, "Heavy load throughput insufficient")
                    self.assertLess(error_rate, 0.05, "Heavy load error rate unacceptable")
                    
                else:  # burst_load
                    self.assertLess(p95_latency, 10.0, "Burst load P95 latency too high")
                    self.assertLess(error_rate, 0.10, "Burst load error rate unacceptable")
                
                # Common assertions for all loads
                self.assertEqual(len(results), num_circuits, "Not all circuits completed")
                self.assertLess(avg_memory_mb, self.performance_thresholds['max_memory_mb'])
                
                print(f"{scenario['name']} Results:")
                print(f"  Circuits: {num_circuits}, Workers: {max_workers}")
                print(f"  Success Rate: {(1-error_rate)*100:.1f}%")
                print(f"  Throughput: {throughput:.2f} circuits/sec")
                print(f"  Avg Latency: {avg_latency:.3f}s")
                print(f"  P95 Latency: {p95_latency:.3f}s")
                print(f"  Avg Memory: {avg_memory_mb:.1f}MB")
    
    @pytest.mark.load
    @pytest.mark.slow
    async def test_quantum_ml_pipeline_throughput_scaling(self):
        """Test throughput scaling of quantum ML training pipeline."""
        scaling_scenarios = [
            {'batch_size': 10, 'concurrent_batches': 1, 'name': 'single_batch'},
            {'batch_size': 20, 'concurrent_batches': 2, 'name': 'dual_batch'},
            {'batch_size': 25, 'concurrent_batches': 4, 'name': 'quad_batch'},
            {'batch_size': 30, 'concurrent_batches': 8, 'name': 'octa_batch'}
        ]
        
        # Create base pipeline
        def simple_circuit(params, x):
            return float(np.sum(np.sin(params) * np.cos(x)))
        
        pipeline = QuantumMLPipeline(
            circuit=simple_circuit,
            n_qubits=4,
            device=QuantumDevice.SIMULATOR
        )
        
        baseline_throughput = None
        
        for scenario in scaling_scenarios:
            with self.subTest(scaling=scenario['name']):
                batch_size = scenario['batch_size']
                concurrent_batches = scenario['concurrent_batches']
                
                start_time = time.time()
                
                async def process_batch(batch_id):
                    """Process a single batch of quantum ML operations."""
                    try:
                        batch_start = time.time()
                        
                        # Generate batch data
                        X_batch = np.random.random((batch_size, 4))
                        y_batch = np.random.randint(0, 2, batch_size)
                        
                        # Simulate quantum ML processing
                        model = pipeline.train(
                            X_batch, y_batch, 
                            epochs=3,  # Reduced for load testing
                            learning_rate=0.1
                        )
                        
                        # Test model predictions
                        predictions = model.predict(X_batch)
                        
                        batch_time = time.time() - batch_start
                        
                        return {
                            'batch_id': batch_id,
                            'batch_size': batch_size,
                            'execution_time': batch_time,
                            'samples_processed': len(predictions),
                            'success': True
                        }
                        
                    except Exception as e:
                        return {
                            'batch_id': batch_id,
                            'batch_size': batch_size,
                            'execution_time': time.time() - batch_start,
                            'error': str(e),
                            'success': False
                        }
                
                # Execute concurrent batches
                tasks = []
                for batch_id in range(concurrent_batches):
                    task = asyncio.create_task(process_batch(batch_id))
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_time = time.time() - start_time
                
                # Analyze scaling performance
                successful_batches = sum(1 for r in batch_results if isinstance(r, dict) and r.get('success', False))
                total_samples = sum(r.get('samples_processed', 0) for r in batch_results if isinstance(r, dict) and r.get('success', False))
                
                overall_throughput = total_samples / total_time
                batch_throughput = successful_batches / total_time
                
                # Scaling assertions
                if concurrent_batches == 1:
                    baseline_throughput = overall_throughput
                    self.assertGreater(overall_throughput, 2.0, "Single batch throughput too low")
                else:
                    # Should scale reasonably (at least 50% of linear scaling)
                    if baseline_throughput:
                        expected_min_throughput = baseline_throughput * concurrent_batches * 0.5
                        self.assertGreater(overall_throughput, expected_min_throughput,
                                         f"Throughput doesn't scale: {overall_throughput:.2f} < {expected_min_throughput:.2f}")
                
                # Quality assertions
                self.assertGreaterEqual(successful_batches, concurrent_batches * 0.8,
                                      "Too many batch failures")
                
                print(f"{scenario['name']} Results:")
                print(f"  Concurrent Batches: {concurrent_batches}")
                print(f"  Successful Batches: {successful_batches}/{concurrent_batches}")
                print(f"  Total Samples: {total_samples}")
                print(f"  Overall Throughput: {overall_throughput:.2f} samples/sec")
                print(f"  Batch Throughput: {batch_throughput:.2f} batches/sec")
    
    @pytest.mark.load
    def test_quantum_job_queue_scaling_comprehensive(self):
        """Test quantum job queue under comprehensive high load scenarios."""
        queue_scenarios = [
            {'jobs': 100, 'submit_rate': 10, 'name': 'steady_load'},
            {'jobs': 500, 'submit_rate': 50, 'name': 'high_load'},
            {'jobs': 1000, 'submit_rate': 100, 'name': 'peak_load'},
            {'jobs': 2000, 'submit_rate': 200, 'name': 'burst_load'}
        ]
        
        for scenario in queue_scenarios:
            with self.subTest(queue_scenario=scenario['name']):
                job_count = scenario['jobs']
                submit_rate = scenario['submit_rate']  # jobs per second
                
                # Mock quantum job queue with realistic behavior
                with patch('quantum_mlops.backends.QuantumBackend.submit_job') as mock_submit:
                    # Track queue metrics
                    submitted_jobs = []
                    processing_times = []
                    queue_sizes = []
                    
                    def mock_job_submission(circuits, shots=1024):
                        """Mock job submission with realistic processing time."""
                        job_id = f"job_{len(submitted_jobs)}"
                        submit_time = time.time()
                        
                        # Simulate processing delay based on queue load
                        queue_load = len(submitted_jobs)
                        base_processing_time = 0.1  # Base 100ms
                        load_factor = min(queue_load / 100, 5.0)  # Max 5x slowdown
                        processing_time = base_processing_time * (1 + load_factor)
                        
                        job_info = {
                            'job_id': job_id,
                            'submit_time': submit_time,
                            'processing_time': processing_time,
                            'circuits': len(circuits),
                            'shots': shots,
                            'status': 'queued'
                        }
                        
                        submitted_jobs.append(job_info)
                        processing_times.append(processing_time)
                        queue_sizes.append(len(submitted_jobs))
                        
                        return MagicMock(
                            job_id=job_id,
                            status='queued',
                            created_at=submit_time
                        )
                    
                    mock_submit.side_effect = mock_job_submission
                    
                    # Submit jobs at specified rate
                    start_time = time.time()
                    successful_submissions = 0
                    failed_submissions = 0
                    
                    for job_idx in range(job_count):
                        try:
                            # Create mock circuit data
                            circuits = [{'type': 'test', 'qubits': 4}]
                            
                            # Submit job
                            job = mock_submit(circuits, shots=1024)
                            successful_submissions += 1
                            
                            # Rate limiting
                            if submit_rate > 0:
                                expected_time = start_time + (job_idx + 1) / submit_rate
                                current_time = time.time()
                                if current_time < expected_time:
                                    time.sleep(expected_time - current_time)
                            
                        except Exception as e:
                            failed_submissions += 1
                            print(f"Job submission failed: {e}")
                    
                    total_submission_time = time.time() - start_time
                    
                    # Analyze queue performance
                    actual_submit_rate = successful_submissions / total_submission_time
                    max_queue_size = max(queue_sizes) if queue_sizes else 0
                    avg_processing_time = np.mean(processing_times) if processing_times else 0
                    p95_processing_time = np.percentile(processing_times, 95) if processing_times else 0
                    
                    # Performance assertions based on load
                    if scenario['name'] == 'steady_load':
                        self.assertGreater(actual_submit_rate, submit_rate * 0.8,
                                         "Steady load submit rate too low")
                        self.assertLess(avg_processing_time, 0.5,
                                       "Steady load processing time too high")
                        
                    elif scenario['name'] in ['high_load', 'peak_load']:
                        self.assertGreater(actual_submit_rate, submit_rate * 0.6,
                                         f"{scenario['name']} submit rate significantly degraded")
                        self.assertLess(p95_processing_time, 2.0,
                                       f"{scenario['name']} P95 processing time excessive")
                        
                    else:  # burst_load
                        self.assertGreater(actual_submit_rate, submit_rate * 0.4,
                                         "Burst load completely degraded system")
                        self.assertLess(max_queue_size, job_count * 1.5,
                                       "Queue size grew excessively")
                    
                    # Common assertions
                    success_rate = successful_submissions / job_count
                    self.assertGreater(success_rate, 0.95,
                                     f"Job submission success rate too low: {success_rate:.2%}")
                    
                    print(f"{scenario['name']} Queue Results:")
                    print(f"  Jobs: {job_count}, Target Rate: {submit_rate}/sec")
                    print(f"  Success Rate: {success_rate:.2%}")
                    print(f"  Actual Submit Rate: {actual_submit_rate:.1f}/sec")
                    print(f"  Max Queue Size: {max_queue_size}")
                    print(f"  Avg Processing Time: {avg_processing_time:.3f}s")
                    print(f"  P95 Processing Time: {p95_processing_time:.3f}s")
    
    @pytest.mark.load
    def test_quantum_memory_usage_under_sustained_load(self):
        """Test memory usage during sustained intensive quantum operations."""
        load_duration = 30  # seconds
        operation_interval = 0.1  # seconds between operations
        
        # Create model for sustained testing
        model = self.create_model(n_qubits=5, circuit_type='variational', depth=3)
        
        # Memory monitoring
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self.skipTest("psutil not available for memory monitoring")
        
        memory_snapshots = []
        operation_count = 0
        errors = 0
        
        start_time = time.time()
        
        while time.time() - start_time < load_duration:
            operation_start = time.time()
            
            try:
                # Memory snapshot before operation
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # Perform quantum operation
                X_batch = np.random.random((10, 5))
                predictions = model.predict(X_batch)
                
                # Memory snapshot after operation
                memory_after = process.memory_info().rss / 1024 / 1024
                
                memory_snapshots.append({
                    'time': time.time() - start_time,
                    'memory_mb': memory_after,
                    'memory_delta': memory_after - memory_before,
                    'operation_count': operation_count
                })
                
                operation_count += 1
                
            except Exception as e:
                errors += 1
                print(f"Operation {operation_count} failed: {e}")
            
            # Maintain operation interval
            operation_time = time.time() - operation_start
            if operation_time < operation_interval:
                time.sleep(operation_interval - operation_time)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Analyze memory usage patterns
        memory_values = [snapshot['memory_mb'] for snapshot in memory_snapshots]
        memory_deltas = [snapshot['memory_delta'] for snapshot in memory_snapshots]
        
        peak_memory = max(memory_values) if memory_values else final_memory
        avg_memory = np.mean(memory_values) if memory_values else final_memory
        memory_growth = final_memory - initial_memory
        avg_memory_delta = np.mean([abs(delta) for delta in memory_deltas]) if memory_deltas else 0
        
        # Memory usage assertions
        self.assertLess(peak_memory, self.performance_thresholds['max_memory_mb'],
                       f"Peak memory usage {peak_memory:.1f}MB exceeds threshold")
        
        self.assertLess(memory_growth, 100,  # Max 100MB growth
                       f"Memory growth {memory_growth:.1f}MB indicates potential leak")
        
        self.assertLess(avg_memory_delta, 10,  # Max 10MB average delta
                       f"Average memory delta {avg_memory_delta:.1f}MB too high")
        
        # Operation success assertions
        error_rate = errors / max(operation_count, 1)
        self.assertLess(error_rate, 0.05,
                       f"Error rate {error_rate:.2%} too high under sustained load")
        
        self.assertGreater(operation_count, load_duration / operation_interval * 0.8,
                         "Insufficient operations completed during sustained load")
        
        print(f"Sustained Load Memory Results:")
        print(f"  Duration: {load_duration}s, Operations: {operation_count}")
        print(f"  Initial Memory: {initial_memory:.1f}MB")
        print(f"  Peak Memory: {peak_memory:.1f}MB")
        print(f"  Final Memory: {final_memory:.1f}MB")
        print(f"  Memory Growth: {memory_growth:.1f}MB")
        print(f"  Avg Memory Delta: {avg_memory_delta:.1f}MB")
        print(f"  Error Rate: {error_rate:.2%}")
    
    @pytest.mark.load
    @pytest.mark.slow
    def test_quantum_error_rate_under_extreme_load(self):
        """Test quantum error rates under extreme system load conditions."""
        extreme_load_scenarios = [
            {'circuits': 500, 'workers': 20, 'duration': 60, 'name': 'high_concurrency'},
            {'circuits': 1000, 'workers': 10, 'duration': 30, 'name': 'high_volume'},
            {'circuits': 200, 'workers': 50, 'duration': 45, 'name': 'max_parallelism'}
        ]
        
        for scenario in extreme_load_scenarios:
            with self.subTest(extreme_load=scenario['name']):
                num_circuits = scenario['circuits']
                max_workers = min(scenario['workers'], self.max_workers)
                duration = scenario['duration']
                
                # Create test model
                model = self.create_model(n_qubits=4, circuit_type='basic')
                
                # Error tracking
                total_operations = 0
                successful_operations = 0
                error_types = defaultdict(int)
                response_times = []
                
                def execute_with_monitoring():
                    """Execute circuit with comprehensive error monitoring."""
                    operation_start = time.time()
                    
                    try:
                        # Generate test data
                        X = np.random.random((1, 4))
                        
                        # Execute quantum operation
                        prediction = model.predict(X)
                        
                        response_time = time.time() - operation_start
                        response_times.append(response_time)
                        
                        return {'success': True, 'response_time': response_time, 'result': prediction[0]}
                        
                    except MemoryError as e:
                        error_types['memory_error'] += 1
                        return {'success': False, 'error_type': 'memory_error', 'error': str(e)}
                    except TimeoutError as e:
                        error_types['timeout_error'] += 1
                        return {'success': False, 'error_type': 'timeout_error', 'error': str(e)}
                    except Exception as e:
                        error_types['general_error'] += 1
                        return {'success': False, 'error_type': 'general_error', 'error': str(e)}
                
                # Execute extreme load test
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    # Submit operations continuously for specified duration
                    while time.time() - start_time < duration:
                        if len(futures) < num_circuits:
                            future = executor.submit(execute_with_monitoring)
                            futures.append(future)
                        
                        # Collect completed operations
                        completed_futures = [f for f in futures if f.done()]
                        for future in completed_futures:
                            try:
                                result = future.result()
                                total_operations += 1
                                
                                if result['success']:
                                    successful_operations += 1
                                
                            except Exception as e:
                                total_operations += 1
                                error_types['future_error'] += 1
                        
                        # Remove completed futures
                        futures = [f for f in futures if not f.done()]
                        
                        # Small delay to prevent CPU thrashing
                        time.sleep(0.01)
                    
                    # Wait for remaining futures
                    for future in futures:
                        try:
                            result = future.result(timeout=10)
                            total_operations += 1
                            if result['success']:
                                successful_operations += 1
                        except Exception:
                            total_operations += 1
                            error_types['cleanup_error'] += 1
                
                # Analyze extreme load results
                total_errors = sum(error_types.values())
                error_rate = total_errors / max(total_operations, 1)
                success_rate = successful_operations / max(total_operations, 1)
                
                avg_response_time = np.mean(response_times) if response_times else float('inf')
                p95_response_time = np.percentile(response_times, 95) if response_times else float('inf')
                
                # Extreme load assertions
                self.assertLess(error_rate, 0.15,  # Allow higher error rate under extreme load
                               f"Error rate {error_rate:.2%} too high even for extreme load")
                
                self.assertGreater(success_rate, 0.70,  # At least 70% success under extreme load
                                 f"Success rate {success_rate:.2%} too low under extreme load")
                
                self.assertLess(avg_response_time, 5.0,
                               f"Average response time {avg_response_time:.2f}s too high")
                
                self.assertGreater(total_operations, duration * 2,  # At least 2 ops/second
                                 f"Total operations {total_operations} too low for {duration}s test")
                
                # Check that system doesn't completely fail
                major_error_rate = (error_types['memory_error'] + error_types['timeout_error']) / max(total_operations, 1)
                self.assertLess(major_error_rate, 0.10,
                               f"Major error rate {major_error_rate:.2%} indicates system instability")
                
                print(f"{scenario['name']} Extreme Load Results:")
                print(f"  Duration: {duration}s, Workers: {max_workers}")
                print(f"  Total Operations: {total_operations}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Error Rate: {error_rate:.2%}")
                print(f"  Error Types: {dict(error_types)}")
                print(f"  Avg Response Time: {avg_response_time:.3f}s")
                print(f"  P95 Response Time: {p95_response_time:.3f}s")
    
    @pytest.mark.load
    def test_quantum_api_rate_limiting_comprehensive(self):
        """Test comprehensive API rate limiting for quantum service calls."""
        rate_limiting_scenarios = [
            {'rate_limit': 10, 'test_duration': 10, 'burst_factor': 1.0, 'name': 'low_limit'},
            {'rate_limit': 50, 'test_duration': 15, 'burst_factor': 1.5, 'name': 'medium_limit'},
            {'rate_limit': 100, 'test_duration': 20, 'burst_factor': 2.0, 'name': 'high_limit'},
            {'rate_limit': 200, 'test_duration': 10, 'burst_factor': 3.0, 'name': 'burst_test'}
        ]
        
        for scenario in rate_limiting_scenarios:
            with self.subTest(rate_limit_scenario=scenario['name']):
                rate_limit = scenario['rate_limit']  # requests per minute
                test_duration = scenario['test_duration']  # seconds
                burst_factor = scenario['burst_factor']
                
                # Calculate expected limits
                max_requests_per_second = rate_limit / 60.0
                max_requests_total = int(rate_limit * test_duration / 60.0)
                burst_requests = int(max_requests_total * burst_factor)
                
                # Mock rate-limited API
                request_times = []
                successful_requests = 0
                rate_limited_requests = 0
                
                with patch('quantum_mlops.backends.QuantumBackend.submit_job') as mock_submit:
                    def rate_limited_submit(*args, **kwargs):
                        """Mock API call with rate limiting simulation."""
                        current_time = time.time()
                        request_times.append(current_time)
                        
                        # Count recent requests (last minute)
                        recent_requests = [t for t in request_times if current_time - t <= 60]
                        
                        if len(recent_requests) > rate_limit:
                            raise ConnectionError(f"Rate limit exceeded: {len(recent_requests)} > {rate_limit}")
                        
                        # Simulate processing delay based on current load
                        processing_delay = len(recent_requests) * 0.001  # 1ms per recent request
                        time.sleep(processing_delay)
                        
                        return MagicMock(job_id=f'job_{len(request_times)}', status='completed')
                    
                    mock_submit.side_effect = rate_limited_submit
                    
                    # Test rate limiting behavior
                    start_time = time.time()
                    
                    # Phase 1: Normal load
                    normal_duration = test_duration * 0.6
                    while time.time() - start_time < normal_duration:
                        try:
                            mock_submit()
                            successful_requests += 1
                            
                            # Space requests evenly for normal load
                            time.sleep(60.0 / rate_limit)
                            
                        except ConnectionError:
                            rate_limited_requests += 1
                            time.sleep(0.1)  # Brief backoff
                    
                    # Phase 2: Burst load
                    burst_start = time.time()
                    burst_attempts = 0
                    
                    while time.time() - burst_start < test_duration * 0.4 and burst_attempts < burst_requests:
                        try:
                            mock_submit()
                            successful_requests += 1
                            burst_attempts += 1
                            
                            # Minimal delay for burst testing
                            time.sleep(0.01)
                            
                        except ConnectionError:
                            rate_limited_requests += 1
                            burst_attempts += 1
                            time.sleep(0.05)  # Short backoff during burst
                
                total_requests = successful_requests + rate_limited_requests
                
                # Analyze rate limiting effectiveness
                actual_duration = time.time() - start_time
                actual_rate = successful_requests / actual_duration * 60  # requests per minute
                rate_limit_effectiveness = rate_limited_requests / max(total_requests, 1)
                
                # Rate limiting assertions
                if scenario['name'] in ['low_limit', 'medium_limit']:
                    # Should respect rate limits closely
                    self.assertLess(actual_rate, rate_limit * 1.1,
                                   f"Actual rate {actual_rate:.1f} exceeds limit {rate_limit}")
                    
                    self.assertGreater(rate_limit_effectiveness, 0.0,
                                     "Rate limiting should activate under normal conditions")
                    
                elif scenario['name'] == 'high_limit':
                    # Higher limits should allow more throughput
                    self.assertLess(actual_rate, rate_limit * 1.2,
                                   f"Actual rate {actual_rate:.1f} significantly exceeds high limit")
                    
                else:  # burst_test
                    # Burst test should trigger significant rate limiting
                    self.assertGreater(rate_limit_effectiveness, 0.1,
                                     "Burst test should trigger substantial rate limiting")
                    
                    self.assertLess(actual_rate, rate_limit * 1.5,
                                   "Even burst should not completely bypass rate limits")
                
                # Common assertions
                self.assertGreater(successful_requests, 0,
                                 "Should have some successful requests")
                
                self.assertLessEqual(actual_rate, rate_limit * 2,
                                   "Rate limiting completely ineffective")
                
                print(f"{scenario['name']} Rate Limiting Results:")
                print(f"  Rate Limit: {rate_limit}/min, Test Duration: {test_duration}s")
                print(f"  Total Requests: {total_requests}")
                print(f"  Successful: {successful_requests}")
                print(f"  Rate Limited: {rate_limited_requests}")
                print(f"  Actual Rate: {actual_rate:.1f}/min")
                print(f"  Rate Limit Effectiveness: {rate_limit_effectiveness:.2%}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
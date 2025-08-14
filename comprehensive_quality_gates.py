#!/usr/bin/env python3
"""Comprehensive Quality Gates - Final Testing Suite Across All Generations"""

import sys
import os
import json
import time
import subprocess
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class QualityGateResult:
    """Result of a quality gate test."""
    gate_name: str
    category: str
    passed: bool
    score: float
    execution_time: float
    details: Dict[str, Any]
    error_message: str = ""

@dataclass
class QualityGatesSummary:
    """Overall quality gates summary."""
    total_gates: int
    passed_gates: int
    failed_gates: int
    overall_score: float
    execution_time: float
    categories: Dict[str, Dict[str, Any]]
    critical_failures: List[str]
    recommendations: List[str]

class ComprehensiveQualityGates:
    """Comprehensive quality gates testing framework."""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.results = []
        self.start_time = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - QG - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Comprehensive Quality Gates")
    
    def _run_gate(self, gate_func, gate_name: str, category: str, timeout: int = 60) -> QualityGateResult:
        """Run a single quality gate with timeout and error handling."""
        start_time = time.time()
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(gate_func)
                result = future.result(timeout=timeout)
            
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                passed = result.get('passed', False)
                score = result.get('score', 0.0)
                details = result.get('details', {})
            elif isinstance(result, bool):
                passed = result
                score = 1.0 if passed else 0.0
                details = {}
            else:
                passed = bool(result)
                score = 1.0 if passed else 0.0
                details = {'raw_result': result}
            
            return QualityGateResult(
                gate_name=gate_name,
                category=category,
                passed=passed,
                score=score,
                execution_time=execution_time,
                details=details
            )
            
        except concurrent.futures.TimeoutError:
            return QualityGateResult(
                gate_name=gate_name,
                category=category,
                passed=False,
                score=0.0,
                execution_time=timeout,
                details={},
                error_message=f"Timeout after {timeout} seconds"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return QualityGateResult(
                gate_name=gate_name,
                category=category,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={},
                error_message=str(e)
            )
    
    def gate_project_structure(self) -> Dict[str, Any]:
        """Verify project structure and required files."""
        required_files = [
            'README.md',
            'pyproject.toml',
            'requirements.txt',
            'src/quantum_mlops/__init__.py',
            'src/quantum_mlops/core.py',
            'tests/conftest.py'
        ]
        
        optional_files = [
            'LICENSE',
            'CONTRIBUTING.md',
            'ARCHITECTURE.md',
            'docker-compose.yml',
            'Dockerfile',
            'Makefile'
        ]
        
        required_count = 0
        optional_count = 0
        
        for file_path in required_files:
            if (self.project_root / file_path).exists():
                required_count += 1
        
        for file_path in optional_files:
            if (self.project_root / file_path).exists():
                optional_count += 1
        
        required_score = required_count / len(required_files)
        optional_score = optional_count / len(optional_files)
        overall_score = (required_score * 0.8) + (optional_score * 0.2)
        
        return {
            'passed': required_score >= 0.8,
            'score': overall_score,
            'details': {
                'required_files': f"{required_count}/{len(required_files)}",
                'optional_files': f"{optional_count}/{len(optional_files)}",
                'missing_required': [f for f in required_files if not (self.project_root / f).exists()],
                'missing_optional': [f for f in optional_files if not (self.project_root / f).exists()]
            }
        }
    
    def gate_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        try:
            # Count Python files and lines of code
            python_files = list(self.project_root.rglob("*.py"))
            total_lines = 0
            total_files = len(python_files)
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    continue
            
            # Simple quality metrics
            avg_lines_per_file = total_lines / total_files if total_files > 0 else 0
            
            # Check for basic code organization
            has_src_dir = (self.project_root / 'src').exists()
            has_tests_dir = (self.project_root / 'tests').exists()
            has_examples = (self.project_root / 'examples').exists()
            
            organization_score = sum([has_src_dir, has_tests_dir, has_examples]) / 3
            
            # Size appropriateness (not too small, not too large per file)
            size_score = 1.0
            if avg_lines_per_file < 50:
                size_score = 0.7  # Files might be too small
            elif avg_lines_per_file > 1000:
                size_score = 0.6  # Files might be too large
            
            overall_score = (organization_score * 0.5) + (size_score * 0.3) + (min(1.0, total_files / 20) * 0.2)
            
            return {
                'passed': overall_score >= 0.7,
                'score': overall_score,
                'details': {
                    'total_python_files': total_files,
                    'total_lines_of_code': total_lines,
                    'avg_lines_per_file': avg_lines_per_file,
                    'has_src_structure': has_src_dir,
                    'has_tests': has_tests_dir,
                    'has_examples': has_examples,
                    'organization_score': organization_score
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_documentation_completeness(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        try:
            doc_score = 0.0
            total_checks = 5
            
            # README.md exists and has content
            readme_path = self.project_root / 'README.md'
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                if len(readme_content) > 1000:  # Substantial content
                    doc_score += 1
            
            # Architecture documentation
            arch_files = ['ARCHITECTURE.md', 'docs/architecture.md', 'DESIGN.md']
            for arch_file in arch_files:
                if (self.project_root / arch_file).exists():
                    doc_score += 1
                    break
            
            # Contributing guidelines
            contrib_files = ['CONTRIBUTING.md', 'docs/CONTRIBUTING.md']
            for contrib_file in contrib_files:
                if (self.project_root / contrib_file).exists():
                    doc_score += 1
                    break
            
            # Examples directory
            if (self.project_root / 'examples').exists():
                example_files = list((self.project_root / 'examples').glob('*.py'))
                if len(example_files) >= 3:
                    doc_score += 1
            
            # Documentation directory
            docs_dirs = ['docs', 'documentation', 'doc']
            for docs_dir in docs_dirs:
                if (self.project_root / docs_dir).exists():
                    doc_files = list((self.project_root / docs_dir).rglob('*.md'))
                    if len(doc_files) >= 2:
                        doc_score += 1
                    break
            
            final_score = doc_score / total_checks
            
            return {
                'passed': final_score >= 0.6,
                'score': final_score,
                'details': {
                    'documentation_score': f"{doc_score}/{total_checks}",
                    'has_readme': readme_path.exists(),
                    'readme_length': len(readme_content) if 'readme_content' in locals() else 0,
                    'has_architecture_docs': any((self.project_root / f).exists() for f in arch_files),
                    'has_contributing': any((self.project_root / f).exists() for f in contrib_files),
                    'example_count': len(example_files) if 'example_files' in locals() else 0
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_generation1_functionality(self) -> Dict[str, Any]:
        """Test Generation 1 basic functionality."""
        try:
            # Check if Generation 1 demo exists and can run
            gen1_files = [
                'gen1_simple_demo.py',
                'generation1_demo.py',
                'simple_demo.py'
            ]
            
            gen1_file = None
            for filename in gen1_files:
                if (self.project_root / filename).exists():
                    gen1_file = self.project_root / filename
                    break
            
            if not gen1_file:
                return {
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'No Generation 1 demo file found'}
                }
            
            # Try to run the Generation 1 demo
            result = subprocess.run(
                [sys.executable, str(gen1_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            output_contains_success = 'GENERATION 1' in result.stdout and ('SUCCESS' in result.stdout or 'COMPLETE' in result.stdout)
            
            score = 1.0 if success and output_contains_success else (0.5 if success else 0.0)
            
            return {
                'passed': success,
                'score': score,
                'details': {
                    'return_code': result.returncode,
                    'found_demo_file': str(gen1_file.name),
                    'output_length': len(result.stdout),
                    'has_success_message': output_contains_success,
                    'stderr_length': len(result.stderr)
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': 'Generation 1 demo timed out'}
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_generation2_robustness(self) -> Dict[str, Any]:
        """Test Generation 2 robustness features."""
        try:
            gen2_files = [
                'gen2_robust_demo.py',
                'generation2_enhancements.py',
                'robust_enhancements.py'
            ]
            
            gen2_file = None
            for filename in gen2_files:
                if (self.project_root / filename).exists():
                    gen2_file = self.project_root / filename
                    break
            
            if not gen2_file:
                return {
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'No Generation 2 demo file found'}
                }
            
            # Try to run the Generation 2 demo
            result = subprocess.run(
                [sys.executable, str(gen2_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            success = result.returncode == 0
            has_robustness_features = any(keyword in result.stdout.lower() for keyword in 
                                        ['error handling', 'validation', 'monitoring', 'robust', 'generation 2'])
            
            # Check if logs directory was created
            logs_created = (self.project_root / 'logs').exists()
            
            score_components = [success, has_robustness_features, logs_created]
            score = sum(score_components) / len(score_components)
            
            return {
                'passed': success,
                'score': score,
                'details': {
                    'return_code': result.returncode,
                    'found_demo_file': str(gen2_file.name),
                    'has_robustness_features': has_robustness_features,
                    'logs_directory_created': logs_created,
                    'output_length': len(result.stdout),
                    'stderr_length': len(result.stderr)
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': 'Generation 2 demo timed out'}
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_generation3_scalability(self) -> Dict[str, Any]:
        """Test Generation 3 scalability features."""
        try:
            gen3_files = [
                'gen3_scale_demo.py',
                'generation3_optimization.py',
                'scaling_optimization.py'
            ]
            
            gen3_file = None
            for filename in gen3_files:
                if (self.project_root / filename).exists():
                    gen3_file = self.project_root / filename
                    break
            
            if not gen3_file:
                return {
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'No Generation 3 demo file found'}
                }
            
            # Try to run the Generation 3 demo
            result = subprocess.run(
                [sys.executable, str(gen3_file)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=90
            )
            
            success = result.returncode == 0
            has_scaling_features = any(keyword in result.stdout.lower() for keyword in 
                                     ['parallel', 'cache', 'throughput', 'scalab', 'generation 3'])
            
            # Look for performance metrics in output
            has_performance_metrics = any(keyword in result.stdout.lower() for keyword in
                                        ['samples/sec', 'efficiency', 'throughput'])
            
            score_components = [success, has_scaling_features, has_performance_metrics]
            score = sum(score_components) / len(score_components)
            
            return {
                'passed': success,
                'score': score,
                'details': {
                    'return_code': result.returncode,
                    'found_demo_file': str(gen3_file.name),
                    'has_scaling_features': has_scaling_features,
                    'has_performance_metrics': has_performance_metrics,
                    'output_length': len(result.stdout),
                    'stderr_length': len(result.stderr)
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': 'Generation 3 demo timed out'}
            }
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_deployment_readiness(self) -> Dict[str, Any]:
        """Check deployment readiness."""
        deployment_files = [
            'Dockerfile',
            'docker-compose.yml',
            'k8s/deployment.yaml',
            'terraform/main.tf',
            'requirements.txt',
            'pyproject.toml'
        ]
        
        security_files = [
            'SECURITY.md',
            'src/quantum_mlops/security',
            '.gitignore'
        ]
        
        deployment_score = 0
        security_score = 0
        
        for file_path in deployment_files:
            if (self.project_root / file_path).exists():
                deployment_score += 1
        
        for file_path in security_files:
            if (self.project_root / file_path).exists():
                security_score += 1
        
        deployment_ratio = deployment_score / len(deployment_files)
        security_ratio = security_score / len(security_files)
        
        overall_score = (deployment_ratio * 0.7) + (security_ratio * 0.3)
        
        return {
            'passed': overall_score >= 0.6,
            'score': overall_score,
            'details': {
                'deployment_files_present': f"{deployment_score}/{len(deployment_files)}",
                'security_files_present': f"{security_score}/{len(security_files)}",
                'deployment_score': deployment_ratio,
                'security_score': security_ratio,
                'missing_deployment': [f for f in deployment_files if not (self.project_root / f).exists()],
                'missing_security': [f for f in security_files if not (self.project_root / f).exists()]
            }
        }
    
    def gate_integration_completeness(self) -> Dict[str, Any]:
        """Check integration and example completeness."""
        try:
            # Check examples directory
            examples_dir = self.project_root / 'examples'
            if not examples_dir.exists():
                return {
                    'passed': False,
                    'score': 0.0,
                    'details': {'error': 'No examples directory found'}
                }
            
            example_files = list(examples_dir.glob('*.py'))
            
            # Check for different types of examples
            integration_examples = [f for f in example_files if 'integration' in f.name.lower()]
            backend_examples = [f for f in example_files if 'backend' in f.name.lower()]
            monitoring_examples = [f for f in example_files if 'monitoring' in f.name.lower()]
            
            example_score = min(1.0, len(example_files) / 5)  # Expect at least 5 examples
            diversity_score = len(set([
                bool(integration_examples),
                bool(backend_examples), 
                bool(monitoring_examples)
            ])) / 3
            
            overall_score = (example_score * 0.6) + (diversity_score * 0.4)
            
            return {
                'passed': overall_score >= 0.5,
                'score': overall_score,
                'details': {
                    'total_examples': len(example_files),
                    'integration_examples': len(integration_examples),
                    'backend_examples': len(backend_examples),
                    'monitoring_examples': len(monitoring_examples),
                    'example_score': example_score,
                    'diversity_score': diversity_score
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def gate_performance_benchmarks(self) -> Dict[str, Any]:
        """Check performance benchmark results."""
        try:
            # Look for performance results files
            result_files = [
                'generation1_results.json',
                'generation2_robust_results.json', 
                'generation3_scalable_results.json'
            ]
            
            results_found = 0
            performance_metrics = {}
            
            for result_file in result_files:
                file_path = self.project_root / result_file
                if file_path.exists():
                    results_found += 1
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Extract performance metrics if available
                        if 'performance_metrics' in data:
                            performance_metrics[result_file] = data['performance_metrics']
                        elif 'results' in data:
                            performance_metrics[result_file] = data['results']
                    except:
                        pass
            
            results_score = results_found / len(result_files)
            
            # Check if we have meaningful performance data
            has_perf_data = len(performance_metrics) > 0
            perf_data_score = 1.0 if has_perf_data else 0.0
            
            overall_score = (results_score * 0.7) + (perf_data_score * 0.3)
            
            return {
                'passed': overall_score >= 0.5,
                'score': overall_score,
                'details': {
                    'results_files_found': f"{results_found}/{len(result_files)}",
                    'has_performance_data': has_perf_data,
                    'performance_metrics_count': len(performance_metrics),
                    'results_score': results_score
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def run_all_gates(self) -> QualityGatesSummary:
        """Run all quality gates and return comprehensive summary."""
        self.logger.info("Starting comprehensive quality gates execution")
        
        # Define all gates
        gates = [
            (self.gate_project_structure, "Project Structure", "Foundation"),
            (self.gate_code_quality, "Code Quality", "Foundation"),
            (self.gate_documentation_completeness, "Documentation", "Foundation"),
            (self.gate_generation1_functionality, "Generation 1 Functionality", "Core Features"),
            (self.gate_generation2_robustness, "Generation 2 Robustness", "Core Features"), 
            (self.gate_generation3_scalability, "Generation 3 Scalability", "Core Features"),
            (self.gate_deployment_readiness, "Deployment Readiness", "Production"),
            (self.gate_integration_completeness, "Integration Examples", "Production"),
            (self.gate_performance_benchmarks, "Performance Benchmarks", "Production")
        ]
        
        # Run all gates
        self.results = []
        for gate_func, gate_name, category in gates:
            self.logger.info(f"Running gate: {gate_name}")
            result = self._run_gate(gate_func, gate_name, category)
            self.results.append(result)
            
            if result.passed:
                self.logger.info(f"✅ {gate_name}: PASSED (Score: {result.score:.2f})")
            else:
                self.logger.warning(f"❌ {gate_name}: FAILED (Score: {result.score:.2f}) - {result.error_message}")
        
        # Calculate summary statistics
        total_gates = len(self.results)
        passed_gates = sum(1 for r in self.results if r.passed)
        failed_gates = total_gates - passed_gates
        overall_score = sum(r.score for r in self.results) / total_gates if total_gates > 0 else 0.0
        total_execution_time = time.time() - self.start_time
        
        # Group by categories
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {'gates': [], 'passed': 0, 'total': 0, 'score': 0.0}
            
            categories[result.category]['gates'].append(result.gate_name)
            categories[result.category]['total'] += 1
            categories[result.category]['score'] += result.score
            
            if result.passed:
                categories[result.category]['passed'] += 1
        
        # Calculate category averages
        for category in categories:
            cat_total = categories[category]['total']
            categories[category]['avg_score'] = categories[category]['score'] / cat_total if cat_total > 0 else 0.0
            categories[category]['pass_rate'] = categories[category]['passed'] / cat_total if cat_total > 0 else 0.0
        
        # Identify critical failures
        critical_failures = [r.gate_name for r in self.results if not r.passed and r.category in ['Foundation', 'Core Features']]
        
        # Generate recommendations
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("CRITICAL: Overall quality score is below 50%. Immediate attention required.")
        elif overall_score < 0.7:
            recommendations.append("WARNING: Overall quality score is below 70%. Consider improvements.")
        
        foundation_score = categories.get('Foundation', {}).get('avg_score', 0.0)
        if foundation_score < 0.7:
            recommendations.append("Foundation issues detected: Improve project structure, code quality, or documentation.")
        
        core_features_score = categories.get('Core Features', {}).get('avg_score', 0.0)
        if core_features_score < 0.8:
            recommendations.append("Core features need attention: Ensure all generation demos work correctly.")
        
        production_score = categories.get('Production', {}).get('avg_score', 0.0)
        if production_score < 0.6:
            recommendations.append("Production readiness is low: Improve deployment configuration and examples.")
        
        if len(critical_failures) > 2:
            recommendations.append("Multiple critical failures detected. Focus on foundation and core features first.")
        
        if not recommendations:
            if overall_score >= 0.9:
                recommendations.append("Excellent! All quality gates performing well. Ready for production.")
            else:
                recommendations.append("Good quality overall. Continue monitoring and improving.")
        
        summary = QualityGatesSummary(
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            overall_score=overall_score,
            execution_time=total_execution_time,
            categories=categories,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
        
        self.logger.info(f"Quality gates completed: {passed_gates}/{total_gates} passed ({overall_score:.1%} overall)")
        
        return summary
    
    def generate_report(self, summary: QualityGatesSummary) -> str:
        """Generate detailed quality gates report."""
        report = []
        
        report.append("🛡️ COMPREHENSIVE QUALITY GATES REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append(f"Execution Time: {summary.execution_time:.2f} seconds")
        report.append("")
        
        # Overall summary
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Gates: {summary.total_gates}")
        report.append(f"Passed: {summary.passed_gates} ✅")
        report.append(f"Failed: {summary.failed_gates} ❌")
        report.append(f"Overall Score: {summary.overall_score:.1%}")
        report.append("")
        
        # Category breakdown
        report.append("🏷️ CATEGORY BREAKDOWN")
        report.append("-" * 40)
        for category, stats in summary.categories.items():
            report.append(f"{category}:")
            report.append(f"  Pass Rate: {stats['pass_rate']:.1%} ({stats['passed']}/{stats['total']})")
            report.append(f"  Avg Score: {stats['avg_score']:.1%}")
            report.append(f"  Gates: {', '.join(stats['gates'])}")
            report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 40)
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"{status} | {result.gate_name} ({result.category})")
            report.append(f"    Score: {result.score:.2f}")
            report.append(f"    Time: {result.execution_time:.3f}s")
            
            if result.error_message:
                report.append(f"    Error: {result.error_message}")
            
            if result.details:
                for key, value in result.details.items():
                    if key != 'error':
                        report.append(f"    {key}: {value}")
            
            report.append("")
        
        # Critical failures
        if summary.critical_failures:
            report.append("🚨 CRITICAL FAILURES")
            report.append("-" * 40)
            for failure in summary.critical_failures:
                report.append(f"• {failure}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(summary.recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Final assessment
        if summary.overall_score >= 0.9:
            assessment = "🌟 EXCELLENT - Ready for production deployment"
        elif summary.overall_score >= 0.8:
            assessment = "✅ GOOD - Minor improvements recommended"
        elif summary.overall_score >= 0.7:
            assessment = "⚠️ ACCEPTABLE - Several improvements needed"
        elif summary.overall_score >= 0.5:
            assessment = "❌ NEEDS WORK - Major improvements required"
        else:
            assessment = "🚨 CRITICAL - Significant issues must be addressed"
        
        report.append("🎯 FINAL ASSESSMENT")
        report.append("-" * 40)
        report.append(assessment)
        report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function."""
    print("🚀 QUANTUM MLOPS WORKBENCH - COMPREHENSIVE QUALITY GATES")
    print("=" * 80)
    print("🛡️ Running Final Quality Assurance Across All Generations")
    print()
    
    try:
        # Initialize quality gates
        quality_gates = ComprehensiveQualityGates()
        
        # Run all gates
        summary = quality_gates.run_all_gates()
        
        # Generate report
        report = quality_gates.generate_report(summary)
        
        # Print report
        print(report)
        
        # Save detailed results
        results_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': asdict(summary),
            'detailed_results': [asdict(r) for r in quality_gates.results]
        }
        
        with open('comprehensive_quality_gates_report.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print("💾 Detailed results saved to: comprehensive_quality_gates_report.json")
        print()
        
        # Determine overall success
        if summary.overall_score >= 0.8 and len(summary.critical_failures) == 0:
            print("🌟 COMPREHENSIVE QUALITY GATES: FULL SUCCESS!")
            return 0
        elif summary.overall_score >= 0.7:
            print("✅ COMPREHENSIVE QUALITY GATES: SUCCESS!")
            return 0
        elif summary.overall_score >= 0.5:
            print("⚠️ COMPREHENSIVE QUALITY GATES: PARTIAL SUCCESS")
            return 1
        else:
            print("❌ COMPREHENSIVE QUALITY GATES: NEEDS SIGNIFICANT IMPROVEMENT")
            return 2
    
    except Exception as e:
        print(f"\n💥 QUALITY GATES EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())
"""Quantum Advantage Analysis Engine.

This module provides a unified interface for comprehensive quantum advantage detection
across multiple algorithms and metrics, with automated analysis and reporting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from datetime import datetime
import json

from .quantum_kernel_advantage import QuantumKernelAnalyzer, KernelAdvantageResult
from .variational_advantage_protocols import VariationalAdvantageAnalyzer, VariationalAdvantageResult
from .noise_resilient_testing import NoiseResilientTester, NoiseAdvantageResult
from .multi_metric_supremacy import QuantumSupremacyAnalyzer, SupremacyResult

from ..logging_config import get_logger
from ..exceptions import QuantumMLOpsException

logger = get_logger(__name__)


class AnalysisType(Enum):
    """Types of quantum advantage analysis."""
    
    KERNEL_ADVANTAGE = "kernel_advantage"
    VARIATIONAL_ADVANTAGE = "variational_advantage"
    NOISE_RESILIENT = "noise_resilient"
    QUANTUM_SUPREMACY = "quantum_supremacy"
    COMPREHENSIVE = "comprehensive"


class AdvantageConfidence(Enum):
    """Confidence levels for advantage assessment."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ComprehensiveAdvantageResult:
    """Comprehensive quantum advantage analysis results."""
    
    # Analysis metadata
    analysis_timestamp: str
    analysis_types: List[str]
    problem_characteristics: Dict[str, Any]
    
    # Individual analysis results
    kernel_advantage: Optional[KernelAdvantageResult] = None
    variational_advantage: Optional[VariationalAdvantageResult] = None
    noise_resilient_advantage: Optional[NoiseAdvantageResult] = None
    quantum_supremacy: Optional[SupremacyResult] = None
    
    # Unified advantage metrics
    overall_advantage_score: float = 0.0
    advantage_confidence: AdvantageConfidence = AdvantageConfidence.NONE
    advantage_category: str = "none"
    
    # Recommendations and insights
    key_advantages: List[str] = None
    limitations: List[str] = None
    recommendations: List[str] = None
    
    # Performance summary
    performance_summary: Dict[str, float] = None
    statistical_significance: bool = False
    
    # Resource analysis
    resource_requirements: Dict[str, Any] = None
    practical_feasibility: str = "unknown"


class AdvantageAnalysisEngine:
    """Unified quantum advantage analysis engine."""
    
    def __init__(
        self,
        n_qubits: int,
        enable_kernel_analysis: bool = True,
        enable_variational_analysis: bool = True,
        enable_noise_analysis: bool = True,
        enable_supremacy_analysis: bool = True,
        shots: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """Initialize advantage analysis engine.
        
        Args:
            n_qubits: Number of qubits for analysis
            enable_kernel_analysis: Enable kernel advantage analysis
            enable_variational_analysis: Enable variational advantage analysis
            enable_noise_analysis: Enable noise-resilient analysis
            enable_supremacy_analysis: Enable supremacy analysis
            shots: Number of measurement shots
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.shots = shots
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize analyzers
        self.analyzers = {}
        
        if enable_kernel_analysis:
            self.analyzers['kernel'] = QuantumKernelAnalyzer(
                n_qubits=n_qubits,
                shots=shots,
                seed=seed
            )
        
        if enable_variational_analysis:
            self.analyzers['variational'] = VariationalAdvantageAnalyzer(
                n_qubits=n_qubits,
                shots=shots,
                seed=seed
            )
        
        if enable_noise_analysis:
            self.analyzers['noise'] = NoiseResilientTester(
                n_qubits=n_qubits,
                shots=shots,
                seed=seed
            )
        
        if enable_supremacy_analysis:
            self.analyzers['supremacy'] = QuantumSupremacyAnalyzer(
                max_qubits=n_qubits,
                shots=shots,
                seed=seed
            )
        
        logger.info(
            f"Initialized AdvantageAnalysisEngine with {n_qubits} qubits, "
            f"enabled analyzers: {list(self.analyzers.keys())}"
        )
    
    def analyze_problem_characteristics(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        problem_type: str = "unknown"
    ) -> Dict[str, Any]:
        """Analyze problem characteristics to guide advantage analysis."""
        
        characteristics = {
            "problem_type": problem_type,
            "n_qubits": self.n_qubits,
            "timestamp": datetime.now().isoformat()
        }
        
        if X is not None:
            characteristics.update({
                "n_samples": X.shape[0],
                "n_features": X.shape[1] if len(X.shape) > 1 else 1,
                "feature_range": (float(np.min(X)), float(np.max(X))),
                "feature_variance": float(np.var(X))
            })
        
        if y is not None:
            if len(np.unique(y)) <= 10:  # Likely classification
                characteristics.update({
                    "task_type": "classification",
                    "n_classes": len(np.unique(y)),
                    "class_distribution": {str(k): int(v) for k, v in 
                                         zip(*np.unique(y, return_counts=True))}
                })
            else:  # Likely regression
                characteristics.update({
                    "task_type": "regression",
                    "target_range": (float(np.min(y)), float(np.max(y))),
                    "target_variance": float(np.var(y))
                })
        
        return characteristics
    
    def run_kernel_advantage_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs: Any
    ) -> KernelAdvantageResult:
        """Run quantum kernel advantage analysis."""
        
        if 'kernel' not in self.analyzers:
            raise QuantumMLOpsException("Kernel analyzer not enabled")
        
        logger.info("Running kernel advantage analysis")
        
        analyzer = self.analyzers['kernel']
        result = analyzer.comprehensive_advantage_analysis(X, y, **kwargs)
        
        return result
    
    def run_variational_advantage_analysis(
        self,
        cost_function: Callable,
        **kwargs: Any
    ) -> VariationalAdvantageResult:
        """Run variational quantum advantage analysis."""
        
        if 'variational' not in self.analyzers:
            raise QuantumMLOpsException("Variational analyzer not enabled")
        
        logger.info("Running variational advantage analysis")
        
        analyzer = self.analyzers['variational']
        result = analyzer.comprehensive_advantage_analysis(cost_function, **kwargs)
        
        return result
    
    def run_noise_resilient_analysis(
        self,
        quantum_circuit: Callable,
        classical_model: Any,
        dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any
    ) -> NoiseAdvantageResult:
        """Run noise-resilient advantage analysis."""
        
        if 'noise' not in self.analyzers:
            raise QuantumMLOpsException("Noise analyzer not enabled")
        
        logger.info("Running noise-resilient advantage analysis")
        
        analyzer = self.analyzers['noise']
        result = analyzer.comprehensive_noise_advantage_analysis(
            quantum_circuit, classical_model, dataset=dataset, **kwargs
        )
        
        return result
    
    def run_supremacy_analysis(
        self,
        problem_sizes: Optional[List[int]] = None,
        **kwargs: Any
    ) -> SupremacyResult:
        """Run quantum supremacy analysis."""
        
        if 'supremacy' not in self.analyzers:
            raise QuantumMLOpsException("Supremacy analyzer not enabled")
        
        logger.info("Running quantum supremacy analysis")
        
        analyzer = self.analyzers['supremacy']
        result = analyzer.comprehensive_supremacy_analysis(
            problem_sizes=problem_sizes, **kwargs
        )
        
        return result
    
    def comprehensive_analysis(
        self,
        analysis_config: Dict[str, Any],
        **kwargs: Any
    ) -> ComprehensiveAdvantageResult:
        """Run comprehensive quantum advantage analysis."""
        
        logger.info("Starting comprehensive quantum advantage analysis")
        
        # Extract analysis parameters
        X = analysis_config.get('X')
        y = analysis_config.get('y')
        cost_function = analysis_config.get('cost_function')
        quantum_circuit = analysis_config.get('quantum_circuit')
        classical_model = analysis_config.get('classical_model')
        problem_sizes = analysis_config.get('problem_sizes')
        analysis_types = analysis_config.get(
            'analysis_types', 
            list(self.analyzers.keys())
        )
        
        # Analyze problem characteristics
        problem_characteristics = self.analyze_problem_characteristics(
            X, y, analysis_config.get('problem_type', 'unknown')
        )
        
        # Initialize result
        result = ComprehensiveAdvantageResult(
            analysis_timestamp=datetime.now().isoformat(),
            analysis_types=analysis_types,
            problem_characteristics=problem_characteristics
        )
        
        # Run individual analyses
        individual_scores = []
        
        if 'kernel' in analysis_types and X is not None and y is not None:
            try:
                result.kernel_advantage = self.run_kernel_advantage_analysis(X, y)
                individual_scores.append(result.kernel_advantage.overall_advantage_score)
            except Exception as e:
                logger.warning(f"Kernel analysis failed: {e}")
        
        if 'variational' in analysis_types and cost_function is not None:
            try:
                result.variational_advantage = self.run_variational_advantage_analysis(
                    cost_function
                )
                individual_scores.append(result.variational_advantage.overall_advantage_score)
            except Exception as e:
                logger.warning(f"Variational analysis failed: {e}")
        
        if 'noise' in analysis_types and quantum_circuit is not None and classical_model is not None:
            try:
                dataset = (X, y) if X is not None and y is not None else None
                result.noise_resilient_advantage = self.run_noise_resilient_analysis(
                    quantum_circuit, classical_model, dataset
                )
                individual_scores.append(result.noise_resilient_advantage.noise_resilient_advantage_score)
            except Exception as e:
                logger.warning(f"Noise analysis failed: {e}")
        
        if 'supremacy' in analysis_types:
            try:
                result.quantum_supremacy = self.run_supremacy_analysis(problem_sizes)
                individual_scores.append(result.quantum_supremacy.supremacy_confidence)
            except Exception as e:
                logger.warning(f"Supremacy analysis failed: {e}")
        
        # Compute overall metrics
        result.overall_advantage_score = np.mean(individual_scores) if individual_scores else 0.0
        
        # Determine advantage confidence
        if result.overall_advantage_score > 0.7:
            result.advantage_confidence = AdvantageConfidence.HIGH
            result.advantage_category = "strong_quantum_advantage"
        elif result.overall_advantage_score > 0.5:
            result.advantage_confidence = AdvantageConfidence.MEDIUM
            result.advantage_category = "moderate_quantum_advantage"
        elif result.overall_advantage_score > 0.3:
            result.advantage_confidence = AdvantageConfidence.LOW
            result.advantage_category = "weak_quantum_advantage"
        else:
            result.advantage_confidence = AdvantageConfidence.NONE
            result.advantage_category = "no_quantum_advantage"
        
        # Generate insights and recommendations
        result.key_advantages = self._extract_key_advantages(result)
        result.limitations = self._extract_limitations(result)
        result.recommendations = self._generate_recommendations(result)
        
        # Performance summary
        result.performance_summary = self._generate_performance_summary(result)
        
        # Statistical significance
        result.statistical_significance = self._assess_statistical_significance(result)
        
        # Resource analysis
        result.resource_requirements = self._analyze_resource_requirements(result)
        result.practical_feasibility = self._assess_practical_feasibility(result)
        
        logger.info(
            f"Comprehensive analysis complete. "
            f"Overall advantage: {result.advantage_category}"
        )
        
        return result
    
    def _extract_key_advantages(
        self,
        result: ComprehensiveAdvantageResult
    ) -> List[str]:
        """Extract key quantum advantages from analysis results."""
        
        advantages = []
        
        if result.kernel_advantage:
            if result.kernel_advantage.spectral_advantage > 0.1:
                advantages.append("Strong quantum kernel spectral advantage")
            if result.kernel_advantage.expressivity_advantage > 0.05:
                advantages.append("Enhanced quantum feature map expressivity")
        
        if result.variational_advantage:
            if result.variational_advantage.landscape_advantage > 0.1:
                advantages.append("Favorable quantum optimization landscape")
            if not result.variational_advantage.plateau_detected:
                advantages.append("Absence of barren plateau")
        
        if result.noise_resilient_advantage:
            if result.noise_resilient_advantage.advantage_lost_threshold > 0.05:
                advantages.append("Good noise resilience")
            if result.noise_resilient_advantage.mitigation_improvement > 0.1:
                advantages.append("Effective error mitigation")
        
        if result.quantum_supremacy:
            if result.quantum_supremacy.scaling_advantage > 0.5:
                advantages.append("Superior quantum scaling")
            if result.quantum_supremacy.sample_efficiency_advantage > 2.0:
                advantages.append("Better sample complexity")
        
        return advantages or ["No significant quantum advantages detected"]
    
    def _extract_limitations(
        self,
        result: ComprehensiveAdvantageResult
    ) -> List[str]:
        """Extract limitations and challenges."""
        
        limitations = []
        
        if result.variational_advantage and result.variational_advantage.plateau_detected:
            limitations.append("Barren plateau detected")
        
        if result.noise_resilient_advantage:
            if result.noise_resilient_advantage.advantage_lost_threshold < 0.01:
                limitations.append("Low noise threshold")
        
        if result.quantum_supremacy and not result.quantum_supremacy.supremacy_achieved:
            limitations.append("Quantum supremacy not demonstrated")
        
        if result.overall_advantage_score < 0.3:
            limitations.append("Overall quantum advantage is weak")
        
        return limitations or ["No significant limitations identified"]
    
    def _generate_recommendations(
        self,
        result: ComprehensiveAdvantageResult
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        if result.advantage_confidence == AdvantageConfidence.HIGH:
            recommendations.append("Proceed with quantum implementation")
            recommendations.append("Consider real quantum hardware deployment")
        elif result.advantage_confidence == AdvantageConfidence.MEDIUM:
            recommendations.append("Quantum advantage is promising but requires optimization")
            recommendations.append("Focus on noise mitigation strategies")
        elif result.advantage_confidence == AdvantageConfidence.LOW:
            recommendations.append("Quantum advantage is marginal")
            recommendations.append("Consider hybrid quantum-classical approaches")
        else:
            recommendations.append("Classical methods may be more suitable")
            recommendations.append("Reassess problem formulation for quantum advantage")
        
        # Specific technical recommendations
        if result.variational_advantage and result.variational_advantage.plateau_detected:
            recommendations.append("Use parameter initialization strategies to avoid barren plateaus")
        
        if result.noise_resilient_advantage and result.noise_resilient_advantage.advantage_lost_threshold < 0.02:
            recommendations.append("Implement error mitigation techniques")
            recommendations.append("Consider fault-tolerant quantum computing")
        
        return recommendations
    
    def _generate_performance_summary(
        self,
        result: ComprehensiveAdvantageResult
    ) -> Dict[str, float]:
        """Generate performance summary metrics."""
        
        summary = {
            "overall_advantage_score": result.overall_advantage_score,
            "n_qubits": self.n_qubits
        }
        
        if result.kernel_advantage:
            summary.update({
                "kernel_spectral_advantage": result.kernel_advantage.spectral_advantage,
                "kernel_performance_advantage": result.kernel_advantage.performance_advantage
            })
        
        if result.variational_advantage:
            summary.update({
                "variational_cost_advantage": result.variational_advantage.cost_advantage,
                "variational_expressivity": result.variational_advantage.expressivity_advantage
            })
        
        if result.noise_resilient_advantage:
            summary.update({
                "noise_resilience_score": result.noise_resilient_advantage.noise_resilience_score,
                "noise_threshold": result.noise_resilient_advantage.advantage_lost_threshold
            })
        
        if result.quantum_supremacy:
            summary.update({
                "supremacy_confidence": result.quantum_supremacy.supremacy_confidence,
                "scaling_advantage": result.quantum_supremacy.scaling_advantage
            })
        
        return summary
    
    def _assess_statistical_significance(
        self,
        result: ComprehensiveAdvantageResult
    ) -> bool:
        """Assess overall statistical significance."""
        
        significant_tests = []
        
        if result.kernel_advantage:
            significant_tests.append(result.kernel_advantage.statistically_significant)
        
        if result.variational_advantage:
            significant_tests.append(result.variational_advantage.statistically_significant)
        
        if result.quantum_supremacy:
            significant_tests.append(result.quantum_supremacy.supremacy_p_value < 0.05)
        
        # Require majority of tests to be significant
        return sum(significant_tests) > len(significant_tests) / 2 if significant_tests else False
    
    def _analyze_resource_requirements(
        self,
        result: ComprehensiveAdvantageResult
    ) -> Dict[str, Any]:
        """Analyze resource requirements."""
        
        requirements = {
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "estimated_runtime": "unknown"
        }
        
        if result.quantum_supremacy:
            requirements.update({
                "quantum_resource_scaling": result.quantum_supremacy.quantum_resource_usage,
                "classical_resource_scaling": result.quantum_supremacy.classical_resource_usage
            })
        
        if result.variational_advantage:
            requirements.update({
                "parameter_count": result.variational_advantage.quantum_parameter_count,
                "convergence_steps": result.variational_advantage.quantum_convergence_steps
            })
        
        return requirements
    
    def _assess_practical_feasibility(
        self,
        result: ComprehensiveAdvantageResult
    ) -> str:
        """Assess practical feasibility of quantum implementation."""
        
        if self.n_qubits <= 10 and result.advantage_confidence in [AdvantageConfidence.HIGH, AdvantageConfidence.MEDIUM]:
            return "highly_feasible"
        elif self.n_qubits <= 20 and result.advantage_confidence == AdvantageConfidence.HIGH:
            return "feasible_nisq"
        elif self.n_qubits <= 50:
            return "future_feasible"
        else:
            return "fault_tolerant_required"
    
    def export_results(
        self,
        result: ComprehensiveAdvantageResult,
        filepath: str,
        format: str = "json"
    ) -> None:
        """Export analysis results to file."""
        
        if format == "json":
            # Convert result to JSON-serializable format
            result_dict = {
                "analysis_timestamp": result.analysis_timestamp,
                "analysis_types": result.analysis_types,
                "problem_characteristics": result.problem_characteristics,
                "overall_advantage_score": result.overall_advantage_score,
                "advantage_confidence": result.advantage_confidence.value,
                "advantage_category": result.advantage_category,
                "key_advantages": result.key_advantages,
                "limitations": result.limitations,
                "recommendations": result.recommendations,
                "performance_summary": result.performance_summary,
                "statistical_significance": result.statistical_significance,
                "resource_requirements": result.resource_requirements,
                "practical_feasibility": result.practical_feasibility
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filepath}")
    
    def generate_report(
        self,
        result: ComprehensiveAdvantageResult
    ) -> str:
        """Generate human-readable analysis report."""
        
        report = f"""
# Quantum Advantage Analysis Report

**Analysis Date**: {result.analysis_timestamp}
**Problem Characteristics**: {result.problem_characteristics.get('problem_type', 'Unknown')}
**Qubits**: {self.n_qubits}

## Overall Assessment

**Advantage Score**: {result.overall_advantage_score:.3f}
**Confidence Level**: {result.advantage_confidence.value}
**Category**: {result.advantage_category}
**Statistical Significance**: {'Yes' if result.statistical_significance else 'No'}

## Key Advantages

{chr(10).join(f"- {advantage}" for advantage in result.key_advantages)}

## Limitations

{chr(10).join(f"- {limitation}" for limitation in result.limitations)}

## Recommendations

{chr(10).join(f"- {rec}" for rec in result.recommendations)}

## Technical Details

**Resource Requirements**: {result.resource_requirements}
**Practical Feasibility**: {result.practical_feasibility}

## Performance Summary

{chr(10).join(f"- {k}: {v}" for k, v in result.performance_summary.items())}
"""
        
        return report
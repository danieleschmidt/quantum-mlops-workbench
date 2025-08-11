"""Quantum MLOps Workbench - End-to-end Quantum Machine Learning CI/CD."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "contact@example.com"

from .core import QuantumMLPipeline, QuantumDevice
from .monitoring import QuantumMonitor
from .testing import QuantumTestCase
from .exceptions import (
    QuantumMLOpsException,
    ErrorCategory,
    ErrorSeverity,
    get_error_handler,
    handle_quantum_error,
    safe_execute
)
from .logging_config import get_logger, get_logging_manager
from .health import get_health_monitor, HealthStatus

# New modules
from .integrations import QuantumMLflow, QuantumWandB, setup_experiment_tracking
from .algorithms import VQE, QAOA, create_h2_hamiltonian, create_maxcut_hamiltonian
from .compilation import CircuitOptimizer, OptimizationLevel
from .hyperopt import QuantumHyperOpt, OptimizationResult
from .benchmarking import QuantumAdvantageTester, BenchmarkResult
from .scaling import get_load_balancer, get_job_scheduler, get_auto_scaler, get_performance_optimizer
from .validation import QuantumDataValidator, ValidationResult
from .i18n import get_i18n_manager, SupportedLanguage, set_language, translate

# Advanced quantum advantage detection
from .advantage_detection import (
    AdvantageAnalysisEngine,
    ComprehensiveAdvantageResult,
    QuantumKernelAnalyzer,
    KernelAdvantageResult,
    VariationalAdvantageAnalyzer,
    VariationalAdvantageResult,
    NoiseResilientTester,
    NoiseAdvantageResult,
    QuantumSupremacyAnalyzer,
    SupremacyResult
)

__all__ = [
    "QuantumMLPipeline",
    "QuantumDevice", 
    "QuantumMonitor",
    "QuantumTestCase",
    "QuantumMLOpsException",
    "ErrorCategory",
    "ErrorSeverity",
    "get_error_handler",
    "handle_quantum_error",
    "safe_execute",
    "get_logger",
    "get_logging_manager",
    "get_health_monitor",
    "HealthStatus",
    # New exports
    "QuantumMLflow",
    "QuantumWandB",
    "setup_experiment_tracking",
    "VQE",
    "QAOA",
    "create_h2_hamiltonian",
    "create_maxcut_hamiltonian",
    "CircuitOptimizer",
    "OptimizationLevel",
    "QuantumHyperOpt",
    "OptimizationResult",
    "QuantumAdvantageTester",
    "BenchmarkResult",
    "get_load_balancer",
    "get_job_scheduler",
    "get_auto_scaler",
    "get_performance_optimizer",
    "QuantumDataValidator",
    "ValidationResult",
    "get_i18n_manager",
    "SupportedLanguage",
    "set_language",
    "translate",
    # Advantage detection exports
    "AdvantageAnalysisEngine",
    "ComprehensiveAdvantageResult",
    "QuantumKernelAnalyzer",
    "KernelAdvantageResult",
    "VariationalAdvantageAnalyzer",
    "VariationalAdvantageResult",
    "NoiseResilientTester",
    "NoiseAdvantageResult",
    "QuantumSupremacyAnalyzer",
    "SupremacyResult",
]
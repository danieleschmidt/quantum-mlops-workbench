"""REST API server for quantum MLOps workbench."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import tempfile
import os

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import numpy as np

from .core import QuantumMLPipeline, QuantumDevice, QuantumModel, QuantumMetrics
from .hyperopt import QuantumHyperOpt
from .benchmarking import QuantumAdvantageTester
from .compilation import CircuitOptimizer
from .integrations import setup_experiment_tracking
from .i18n import get_i18n_manager, SupportedLanguage, translate as _
from .exceptions import QuantumMLOpsException
from .health import get_health_monitor, HealthStatus

logger = logging.getLogger(__name__)

if not FASTAPI_AVAILABLE:
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

# Pydantic models for API
class CircuitRequest(BaseModel):
    """Request model for quantum circuit operations."""
    n_qubits: int = Field(..., ge=1, le=30, description="Number of qubits")
    device: str = Field(default="simulator", description="Quantum backend device")
    layers: int = Field(default=3, ge=1, le=10, description="Number of circuit layers")
    entanglement: str = Field(default="linear", description="Entanglement pattern")
    
    @validator('device')
    def validate_device(cls, v):
        valid_devices = [d.value for d in QuantumDevice]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")
        return v


class TrainingRequest(BaseModel):
    """Request model for quantum model training."""
    circuit: CircuitRequest
    training_data: List[List[float]] = Field(..., description="Training features")
    training_labels: List[int] = Field(..., description="Training labels")
    epochs: int = Field(default=50, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(default=0.01, gt=0, lt=1, description="Learning rate")
    track_gradients: bool = Field(default=True, description="Track gradient statistics")
    
    @validator('training_data')
    def validate_training_data(cls, v):
        if len(v) == 0:
            raise ValueError("Training data cannot be empty")
        return v
    
    @validator('training_labels')
    def validate_training_labels(cls, v, values):
        if 'training_data' in values and len(v) != len(values['training_data']):
            raise ValueError("Training labels must match training data length")
        return v


class OptimizationRequest(BaseModel):
    """Request model for hyperparameter optimization."""
    search_space: Dict[str, Any] = Field(..., description="Hyperparameter search space")
    n_trials: int = Field(default=50, ge=1, le=1000, description="Number of optimization trials")
    optimization_backend: str = Field(default="random", description="Optimization backend")
    hardware_budget: int = Field(default=10000, ge=100, description="Quantum hardware budget")


class BenchmarkRequest(BaseModel):
    """Request model for quantum advantage benchmarking."""
    quantum_config: CircuitRequest
    classical_models: List[str] = Field(default=["RandomForest", "SVM"], description="Classical models to compare")
    dataset_size: int = Field(default=100, ge=10, le=10000, description="Dataset size for benchmarking")
    n_runs: int = Field(default=3, ge=1, le=10, description="Number of benchmark runs")


class CompilationRequest(BaseModel):
    """Request model for circuit compilation."""
    circuit: Dict[str, Any] = Field(..., description="Quantum circuit description")
    target_hardware: str = Field(default="simulator", description="Target hardware backend")
    optimization_level: int = Field(default=2, ge=0, le=3, description="Optimization level")


class LanguageRequest(BaseModel):
    """Request model for language setting."""
    language: str = Field(..., description="Language code")
    
    @validator('language')
    def validate_language(cls, v):
        valid_languages = [lang.value for lang in SupportedLanguage]
        if v not in valid_languages:
            raise ValueError(f"Language must be one of: {valid_languages}")
        return v


# API Response models
class TrainingResponse(BaseModel):
    """Response model for training results."""
    model_id: str
    training_time: float
    final_accuracy: float
    final_loss: float
    gradient_variance: float
    message: str


class OptimizationResponse(BaseModel):
    """Response model for optimization results."""
    best_params: Dict[str, Any]
    best_accuracy: float
    n_trials: int
    optimization_time: float
    convergence_achieved: bool
    message: str


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    quantum_accuracy: float
    classical_accuracies: Dict[str, float]
    quantum_advantage: Dict[str, float]
    execution_time: float
    message: str


class CompilationResponse(BaseModel):
    """Response model for compilation results."""
    optimized_circuit: Dict[str, Any]
    gate_reduction: float
    estimated_error: float
    estimated_time: float
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    components: Dict[str, str]
    message: str


# Global storage for models (in production, use proper database)
MODEL_STORAGE: Dict[str, QuantumModel] = {}
OPTIMIZATION_RESULTS: Dict[str, Dict[str, Any]] = {}


def create_api_app() -> FastAPI:
    """Create FastAPI application for quantum MLOps."""
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")
    
    app = FastAPI(
        title="Quantum MLOps Workbench API",
        description="REST API for quantum machine learning operations",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware for cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Security
    security = HTTPBearer()
    
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
        """Verify JWT token (placeholder implementation)."""
        # In production, implement proper JWT validation
        token = credentials.credentials
        if token != "quantum-mlops-token":  # Placeholder validation
            raise HTTPException(status_code=401, detail="Invalid token")
        return token
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check system health status."""
        try:
            health_monitor = get_health_monitor()
            status = health_monitor.check_system_health()
            
            return HealthResponse(
                status=status.status.value,
                timestamp=datetime.now().isoformat(),
                components={comp: info.status.value for comp, info in status.components.items()},
                message=_("health_check", status=status.status.value)
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/language")
    async def set_language(request: LanguageRequest):
        """Set the API language for responses."""
        try:
            language = SupportedLanguage(request.language)
            get_i18n_manager().set_language(language)
            return {"message": _("language_set", language=request.language)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/languages")
    async def get_languages():
        """Get available languages."""
        i18n_manager = get_i18n_manager()
        languages = []
        for lang in i18n_manager.get_available_languages():
            languages.append({
                "code": lang.value,
                "name": i18n_manager.get_language_name(lang)
            })
        return {"languages": languages}
    
    @app.post("/train", response_model=TrainingResponse)
    async def train_quantum_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        token: str = Depends(verify_token)
    ):
        """Train a quantum machine learning model."""
        try:
            # Create quantum circuit function
            def quantum_circuit(params, x):
                # Simplified circuit implementation
                return np.tanh(np.sum(params * x))
            
            # Initialize pipeline
            device = QuantumDevice(request.circuit.device)
            pipeline = QuantumMLPipeline(
                circuit=quantum_circuit,
                n_qubits=request.circuit.n_qubits,
                device=device,
                layers=request.circuit.layers,
                entanglement=request.circuit.entanglement
            )
            
            # Convert data
            X_train = np.array(request.training_data)
            y_train = np.array(request.training_labels)
            
            # Train model
            start_time = datetime.now()
            model = pipeline.train(
                X_train,
                y_train,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
                track_gradients=request.track_gradients
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Store model
            model_id = f"model_{len(MODEL_STORAGE) + 1}_{int(datetime.now().timestamp())}"
            MODEL_STORAGE[model_id] = model
            
            # Extract metrics
            training_history = model.training_history
            final_accuracy = training_history.get('final_accuracy', 0.0)
            final_loss = training_history['loss_history'][-1] if training_history['loss_history'] else 0.0
            gradient_variance = np.mean(training_history.get('gradient_variances', [0.0]))
            
            return TrainingResponse(
                model_id=model_id,
                training_time=training_time,
                final_accuracy=final_accuracy,
                final_loss=final_loss,
                gradient_variance=gradient_variance,
                message=_("training_completed", time=training_time)
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/optimize", response_model=OptimizationResponse)
    async def optimize_hyperparameters(
        request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        token: str = Depends(verify_token)
    ):
        """Optimize hyperparameters for quantum model."""
        try:
            # Initialize optimizer
            optimizer = QuantumHyperOpt(
                search_space=request.search_space,
                optimization_backend=request.optimization_backend,
                hardware_budget=request.hardware_budget
            )
            
            # Generate synthetic training data for optimization
            n_features = request.search_space.get('n_qubits', 4)
            X_train = np.random.rand(100, n_features)
            y_train = np.random.randint(0, 2, 100)
            
            # Define training function
            def train_fn(X, y, **params):
                def quantum_circuit(circuit_params, x):
                    return np.tanh(np.sum(circuit_params * x))
                
                pipeline = QuantumMLPipeline(
                    circuit=quantum_circuit,
                    n_qubits=params.get('n_qubits', 4),
                    device=QuantumDevice.SIMULATOR
                )
                
                return pipeline.train(X, y, epochs=params.get('epochs', 50))
            
            # Run optimization
            start_time = datetime.now()
            results = optimizer.optimize(
                train_fn=train_fn,
                X_train=X_train,
                y_train=y_train,
                n_trials=request.n_trials
            )
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            result_id = f"opt_{len(OPTIMIZATION_RESULTS) + 1}_{int(datetime.now().timestamp())}"
            OPTIMIZATION_RESULTS[result_id] = results
            
            return OptimizationResponse(
                best_params=results['best_params'],
                best_accuracy=results['best_accuracy'],
                n_trials=results['optimization_results'].n_trials,
                optimization_time=optimization_time,
                convergence_achieved=results['optimization_results'].convergence_achieved,
                message=_("optimization_completed", time=optimization_time)
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/benchmark", response_model=BenchmarkResponse)
    async def benchmark_quantum_advantage(
        request: BenchmarkRequest,
        background_tasks: BackgroundTasks,
        token: str = Depends(verify_token)
    ):
        """Benchmark quantum model against classical models."""
        try:
            # Initialize tester
            device = QuantumDevice(request.quantum_config.device)
            tester = QuantumAdvantageTester(device=device)
            
            # Create quantum model
            def quantum_circuit(params, x):
                return np.tanh(np.sum(params * x))
            
            quantum_pipeline = QuantumMLPipeline(
                circuit=quantum_circuit,
                n_qubits=request.quantum_config.n_qubits,
                device=device
            )
            
            # Generate benchmark dataset
            n_features = request.quantum_config.n_qubits
            X_train = np.random.rand(request.dataset_size, n_features)
            y_train = np.random.randint(0, 2, request.dataset_size)
            X_test = np.random.rand(request.dataset_size // 4, n_features)
            y_test = np.random.randint(0, 2, request.dataset_size // 4)
            
            # Create classical models
            classical_models = {}
            if "RandomForest" in request.classical_models:
                from sklearn.ensemble import RandomForestClassifier
                classical_models["RandomForest"] = RandomForestClassifier(n_estimators=10)
            
            if "SVM" in request.classical_models:
                from sklearn.svm import SVC
                classical_models["SVM"] = SVC()
            
            # Run benchmark
            start_time = datetime.now()
            results = tester.compare(
                quantum_model=quantum_pipeline,
                classical_models=classical_models,
                dataset=(X_train, X_test, y_train, y_test),
                n_runs=request.n_runs
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract results
            quantum_accuracy = results['quantum_results'].accuracy
            classical_accuracies = {name: result.accuracy for name, result in results['classical_results'].items()}
            
            # Calculate advantage ratios
            quantum_advantage = {}
            for name, classical_acc in classical_accuracies.items():
                if classical_acc > 0:
                    quantum_advantage[name] = quantum_accuracy / classical_acc
                else:
                    quantum_advantage[name] = 1.0
            
            return BenchmarkResponse(
                quantum_accuracy=quantum_accuracy,
                classical_accuracies=classical_accuracies,
                quantum_advantage=quantum_advantage,
                execution_time=execution_time,
                message=_("benchmark_completed", time=execution_time)
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/compile", response_model=CompilationResponse)
    async def compile_circuit(
        request: CompilationRequest,
        token: str = Depends(verify_token)
    ):
        """Compile quantum circuit for target hardware."""
        try:
            # Initialize compiler
            optimizer = CircuitOptimizer(target_hardware=request.target_hardware)
            
            # Compile circuit
            from .compilation import OptimizationLevel
            opt_level = OptimizationLevel(request.optimization_level)
            
            optimized_circuit = optimizer.compile(
                circuit=request.circuit,
                optimization_level=opt_level,
                preserve_semantics=True
            )
            
            # Get optimization metrics
            report = optimizer.get_optimization_report()
            
            return CompilationResponse(
                optimized_circuit=optimized_circuit,
                gate_reduction=optimizer.gate_reduction,
                estimated_error=report['optimization_metrics'].get('estimated_error', 0.0),
                estimated_time=report['optimization_metrics'].get('estimated_time', 0.0),
                message=_("compilation_completed", hardware=request.target_hardware)
            )
            
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models")
    async def list_models(token: str = Depends(verify_token)):
        """List all trained models."""
        models = []
        for model_id, model in MODEL_STORAGE.items():
            models.append({
                "id": model_id,
                "n_qubits": model.n_qubits,
                "parameter_count": len(model.parameters) if model.parameters is not None else 0,
                "circuit_depth": model.circuit_depth
            })
        return {"models": models}
    
    @app.get("/models/{model_id}")
    async def get_model(model_id: str, token: str = Depends(verify_token)):
        """Get details of a specific model."""
        if model_id not in MODEL_STORAGE:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = MODEL_STORAGE[model_id]
        return {
            "id": model_id,
            "n_qubits": model.n_qubits,
            "parameter_count": len(model.parameters) if model.parameters is not None else 0,
            "circuit_depth": model.circuit_depth,
            "training_history": model.training_history,
            "metadata": model.metadata
        }
    
    @app.delete("/models/{model_id}")
    async def delete_model(model_id: str, token: str = Depends(verify_token)):
        """Delete a trained model."""
        if model_id not in MODEL_STORAGE:
            raise HTTPException(status_code=404, detail="Model not found")
        
        del MODEL_STORAGE[model_id]
        return {"message": f"Model {model_id} deleted successfully"}
    
    @app.get("/status")
    async def get_status():
        """Get system status information."""
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.0",
            "models_stored": len(MODEL_STORAGE),
            "optimizations_stored": len(OPTIMIZATION_RESULTS),
            "supported_devices": [device.value for device in QuantumDevice],
            "current_language": get_i18n_manager().get_language().value
        }
    
    return app


def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
) -> None:
    """Run the FastAPI server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = create_api_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    run_api_server(reload=True)
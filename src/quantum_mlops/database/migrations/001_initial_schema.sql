-- Initial quantum MLOps database schema
-- This file contains the SQL commands to create the initial database schema

-- Enable UUID extension for PostgreSQL (if needed)
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSON DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'active' NOT NULL,
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_id VARCHAR(255) NOT NULL REFERENCES experiments(experiment_id),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'running' NOT NULL,
    config JSON DEFAULT '{}',
    parameters JSON DEFAULT '{}',
    metrics JSON DEFAULT '{}',
    artifacts JSON DEFAULT '[]',
    logs JSON DEFAULT '[]',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    tags JSON DEFAULT '[]',
    file_path VARCHAR(500) NOT NULL,
    n_qubits INTEGER NOT NULL,
    circuit_depth INTEGER,
    parameter_count INTEGER,
    training_metrics JSON DEFAULT '{}',
    training_history JSON DEFAULT '{}',
    validation_metrics JSON DEFAULT '{}',
    metadata JSON DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active' NOT NULL,
    parent_model_id VARCHAR(255) REFERENCES model_registry(model_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    UNIQUE(name, version)
);

-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_id VARCHAR(255) REFERENCES experiments(experiment_id),
    run_id VARCHAR(255) REFERENCES runs(run_id),
    backend VARCHAR(100) NOT NULL,
    device VARCHAR(100),
    status VARCHAR(50) DEFAULT 'queued' NOT NULL,
    priority INTEGER DEFAULT 0,
    n_circuits INTEGER NOT NULL,
    n_qubits INTEGER NOT NULL,
    circuit_depth INTEGER,
    shots INTEGER DEFAULT 1024,
    estimated_cost REAL DEFAULT 0.0,
    actual_cost REAL,
    estimated_runtime REAL,
    actual_runtime REAL,
    queue_time REAL,
    circuit_data JSON,
    parameters JSON DEFAULT '{}',
    results JSON,
    error_message TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Metrics table for time series data
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES runs(run_id),
    metric_name VARCHAR(255) NOT NULL,
    metric_value REAL NOT NULL,
    step INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Deployments table
CREATE TABLE IF NOT EXISTS deployments (
    id SERIAL PRIMARY KEY,
    deployment_id VARCHAR(255) UNIQUE NOT NULL,
    model_id VARCHAR(255) NOT NULL REFERENCES model_registry(model_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active' NOT NULL,
    endpoint_url VARCHAR(500),
    config JSON DEFAULT '{}',
    prediction_count INTEGER DEFAULT 0,
    avg_latency REAL,
    success_rate REAL,
    cpu_limit REAL,
    memory_limit INTEGER,
    gpu_required BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for better performance

-- Experiments indexes
CREATE INDEX IF NOT EXISTS ix_experiments_experiment_id ON experiments(experiment_id);
CREATE INDEX IF NOT EXISTS ix_experiments_name_status ON experiments(name, status);
CREATE INDEX IF NOT EXISTS ix_experiments_created_at ON experiments(created_at);

-- Runs indexes
CREATE INDEX IF NOT EXISTS ix_runs_run_id ON runs(run_id);
CREATE INDEX IF NOT EXISTS ix_runs_experiment_status ON runs(experiment_id, status);
CREATE INDEX IF NOT EXISTS ix_runs_started_at ON runs(started_at);

-- Model registry indexes
CREATE INDEX IF NOT EXISTS ix_models_model_id ON model_registry(model_id);
CREATE INDEX IF NOT EXISTS ix_models_name_status ON model_registry(name, status);
CREATE INDEX IF NOT EXISTS ix_models_n_qubits ON model_registry(n_qubits);
CREATE INDEX IF NOT EXISTS ix_models_created_at ON model_registry(created_at);

-- Jobs indexes
CREATE INDEX IF NOT EXISTS ix_jobs_job_id ON jobs(job_id);
CREATE INDEX IF NOT EXISTS ix_jobs_backend_status ON jobs(backend, status);
CREATE INDEX IF NOT EXISTS ix_jobs_submitted_at ON jobs(submitted_at);
CREATE INDEX IF NOT EXISTS ix_jobs_priority ON jobs(priority);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS ix_metrics_run_name ON metrics(run_id, metric_name);
CREATE INDEX IF NOT EXISTS ix_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS ix_metrics_step ON metrics(step);

-- Deployments indexes
CREATE INDEX IF NOT EXISTS ix_deployments_deployment_id ON deployments(deployment_id);
CREATE INDEX IF NOT EXISTS ix_deployments_model_status ON deployments(model_id, status);
CREATE INDEX IF NOT EXISTS ix_deployments_deployed_at ON deployments(deployed_at);

-- Add comments for documentation
COMMENT ON TABLE experiments IS 'Quantum ML experiments tracking';
COMMENT ON TABLE runs IS 'Individual experiment runs with metrics and artifacts';
COMMENT ON TABLE model_registry IS 'Registry of trained quantum ML models';
COMMENT ON TABLE jobs IS 'Quantum computing job executions';
COMMENT ON TABLE metrics IS 'Time series metrics data for runs';
COMMENT ON TABLE deployments IS 'Model deployment configurations and status';

-- Create a function to update updated_at timestamp (PostgreSQL)
-- This can be adapted for other databases
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at column
CREATE TRIGGER update_experiments_updated_at 
    BEFORE UPDATE ON experiments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_runs_updated_at 
    BEFORE UPDATE ON runs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_registry_updated_at 
    BEFORE UPDATE ON model_registry 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at 
    BEFORE UPDATE ON jobs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_deployments_updated_at 
    BEFORE UPDATE ON deployments 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
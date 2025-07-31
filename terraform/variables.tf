# Terraform variables for Quantum MLOps infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "quantum-mlops"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_group_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "node_group_desired_capacity" {
  description = "Desired capacity for EKS node group"
  type        = number
  default     = 3
}

variable "node_group_max_capacity" {
  description = "Maximum capacity for EKS node group"
  type        = number
  default     = 10
}

variable "node_group_min_capacity" {
  description = "Minimum capacity for EKS node group"
  type        = number
  default     = 1
}

variable "enable_quantum_hardware_nodes" {
  description = "Enable specialized nodes for quantum hardware access"
  type        = bool
  default     = false
}

variable "quantum_hardware_instance_types" {
  description = "Instance types for quantum hardware nodes"
  type        = list(string)
  default     = ["c5.2xlarge", "c5.4xlarge"]
}

variable "enable_monitoring" {
  description = "Enable monitoring stack (Prometheus, Grafana)"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ARN of SSL certificate for load balancer"
  type        = string
  default     = ""
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "quantum-mlops.example.com"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  type        = string
  default     = "quantum-mlops-artifacts"
}

variable "database_instance_class" {
  description = "RDS instance class for MLflow backend"
  type        = string
  default     = "db.t3.micro"
}

variable "database_allocated_storage" {
  description = "Allocated storage for RDS instance (GB)"
  type        = number
  default     = 20
}

variable "backup_retention_period" {
  description = "Backup retention period for RDS (days)"
  type        = number
  default     = 7
}

variable "enable_encryption" {
  description = "Enable encryption at rest for all services"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
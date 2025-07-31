# Terraform outputs for Quantum MLOps infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.quantum_mlops.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.quantum_mlops.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.quantum_mlops.endpoint
  sensitive   = true
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.quantum_mlops.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.quantum_mlops.certificate_authority[0].data
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = aws_eks_cluster.quantum_mlops.vpc_config[0].cluster_security_group_id
}

output "node_groups" {
  description = "EKS node groups"
  value = {
    main = {
      arn           = aws_eks_node_group.main.arn
      status        = aws_eks_node_group.main.status
      capacity_type = aws_eks_node_group.main.capacity_type
    }
  }
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.mlflow.port
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.quantum_mlops.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.quantum_mlops.zone_id
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.quantum_mlops.repository_url
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.quantum_mlops.name
}

output "iam_service_account_role_arn" {
  description = "ARN of the IAM role for service account"
  value       = aws_iam_role.service_account.arn
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.quantum_mlops.name
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

# Kubeconfig command
output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks get-token --cluster-name ${aws_eks_cluster.quantum_mlops.name} --region ${var.aws_region} | kubectl apply -f -"
}

# Connection information
output "connection_info" {
  description = "Connection information for the Quantum MLOps platform"
  value = {
    api_endpoint      = "https://${var.domain_name}"
    mlflow_endpoint   = "https://mlflow.${var.domain_name}"
    grafana_endpoint  = "https://grafana.${var.domain_name}"
    cluster_name      = aws_eks_cluster.quantum_mlops.name
    region           = var.aws_region
  }
}
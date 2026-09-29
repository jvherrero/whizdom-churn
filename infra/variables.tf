variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "vpc_id" {
  type        = string
  description = "ID de la VPC existente"
}

variable "subnet_id" {
  type        = string
  description = "ID de la subnet (pública, si quieres acceder al UI desde fuera)"
}

variable "ami_id" {
  type        = string
  description = "AMI a usar (ej. Amazon Linux 2023 o Ubuntu 22.04)"
}

variable "instance_type" {
  type    = string
  default = "t3.large"
}

variable "key_name" {
  type        = string
  description = "Nombre del key pair existente en AWS para SSH"
}

variable "ssh_allowed_cidr" {
  type        = string
  description = "CIDR permitido para SSH, ej. tu IP/32"
}

variable "mlflow_allowed_cidr" {
  type        = string
  description = "CIDR permitido para acceder al UI de MLflow"
}

variable "s3_bucket_name" {
  type        = string
  description = "Bucket donde MLflow guarda artefactos"
}

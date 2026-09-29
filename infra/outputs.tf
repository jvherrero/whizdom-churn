output "instance_public_ip" {
  value = aws_instance.mlflow_server.public_ip
}

output "instance_id" {
  value = aws_instance.mlflow_server.id
}

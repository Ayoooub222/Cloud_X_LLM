output "instance_id" {
  value = aws_instance.gpu_node.id
}

output "results_bucket" {
  value = aws_s3_bucket.results.id
}

output "mlflow_port_forward" {
  description = "Reach MLflow from the laptop without opening any inbound port."
  value       = "aws ssm start-session --target ${aws_instance.gpu_node.id} --document-name AWS-StartPortForwardingSession --parameters portNumber=5000,localPortNumber=5000"
}

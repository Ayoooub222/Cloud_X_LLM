data "aws_vpc" "default" {
  default = true
}

# No inbound rule: the node is reached through SSM Session Manager (no SSH, no load balancer).
resource "aws_security_group" "node" {
  name_prefix = "${var.project}-node-"
  description = "GPU node, outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "Image pulls, model downloads, S3 and SSM endpoints"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

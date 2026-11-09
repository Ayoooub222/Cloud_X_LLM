data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_instance" "gpu_node" {
  ami                    = data.aws_ami.gpu.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.node.name
  vpc_security_group_ids = [aws_security_group.node.id]

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    k3s_version  = var.k3s_version
    nvdp_version = var.nvdp_version
    repo_url     = var.repo_url
    git_ref      = var.git_ref
    bucket       = aws_s3_bucket.results.id
  })
  user_data_replace_on_change = true

  dynamic "instance_market_options" {
    for_each = var.use_spot ? ["spot"] : []

    content {
      market_type = "spot"

      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  # gp3 baseline: 3000 IOPS, 125 MiB/s (cost model assumption for weight loading)
  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_size
    iops        = 3000
    throughput  = 125
    encrypted   = true
  }

  # IMDSv2 only; hop limit 2 so that pods can reach the instance role
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  tags = {
    Name = "${var.project}-gpu-node"
  }
}

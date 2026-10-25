data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name_prefix        = "${var.project}-node-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

# Least privilege: read/write results only, no delete (versioning keeps history).
data "aws_iam_policy_document" "results_rw" {
  statement {
    sid       = "ListResults"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.results.arn]
  }

  statement {
    sid       = "ReadWriteResults"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.results.arn}/*"]
  }
}

resource "aws_iam_role_policy" "results_rw" {
  name   = "results-rw"
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.results_rw.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "node" {
  name_prefix = "${var.project}-node-"
  role        = aws_iam_role.node.name
}

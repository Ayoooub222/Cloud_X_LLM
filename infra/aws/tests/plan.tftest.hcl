# Offline tests: mocked AWS provider, no credentials, nothing is created.
# Run from infra/aws: terraform init -backend=false && terraform test

mock_provider "aws" {
  mock_data "aws_ami" {
    defaults = {
      id = "ami-0123456789abcdef0"
    }
  }

  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }

  mock_resource "aws_s3_bucket" {
    override_during = plan
    defaults = {
      id  = "cloud-x-llm-results-test"
      arn = "arn:aws:s3:::cloud-x-llm-results-test"
    }
  }
}

variables {
  alert_email = "alerts@example.com"
}

run "defaults_are_on_demand_gp3_imdsv2" {
  command = plan

  assert {
    condition     = aws_instance.gpu_node.instance_type == "g4dn.xlarge"
    error_message = "Default instance type must be g4dn.xlarge (cost model reference)."
  }

  assert {
    condition     = length(aws_instance.gpu_node.instance_market_options) == 0
    error_message = "Spot must be off by default."
  }

  assert {
    condition     = aws_instance.gpu_node.root_block_device[0].volume_type == "gp3" && aws_instance.gpu_node.root_block_device[0].encrypted
    error_message = "Root volume must be encrypted gp3."
  }

  assert {
    condition     = aws_instance.gpu_node.metadata_options[0].http_tokens == "required"
    error_message = "IMDSv2 must be required."
  }
}

run "user_data_deploys_the_default_git_ref" {
  command = plan

  assert {
    condition     = strcontains(aws_instance.gpu_node.user_data, "GIT_REF=\"${var.git_ref}\"")
    error_message = "user_data must clone the git_ref variable."
  }

  assert {
    condition     = strcontains(aws_instance.gpu_node.user_data, "RESULTS_BUCKET=\"cloud-x-llm-results-test\"")
    error_message = "user_data must receive the results bucket name."
  }
}

run "spot_adds_market_options" {
  command = plan

  variables {
    use_spot = true
  }

  assert {
    condition     = aws_instance.gpu_node.instance_market_options[0].market_type == "spot"
    error_message = "use_spot = true must request a spot instance."
  }
}

run "budget_has_three_alerts" {
  command = plan

  assert {
    condition     = length(aws_budgets_budget.monthly.notification) == 3
    error_message = "Expected 50 % actual, 100 % actual and 100 % forecasted alerts."
  }

  assert {
    condition     = aws_budgets_budget.monthly.limit_amount == "10"
    error_message = "Default monthly budget must be 10 USD."
  }
}

run "results_bucket_is_private" {
  command = plan

  assert {
    condition     = aws_s3_bucket_public_access_block.results.block_public_acls && aws_s3_bucket_public_access_block.results.restrict_public_buckets
    error_message = "The results bucket must block public access."
  }
}

run "rejects_unsupported_instance_type" {
  command = plan

  variables {
    instance_type = "p4d.24xlarge"
  }

  expect_failures = [var.instance_type]
}

run "rejects_small_root_volume" {
  command = plan

  variables {
    root_volume_size = 40
  }

  expect_failures = [var.root_volume_size]
}

run "rejects_invalid_alert_email" {
  command = plan

  variables {
    alert_email = "not-an-email"
  }

  expect_failures = [var.alert_email]
}

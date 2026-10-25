variable "project" {
  description = "Name prefix of every resource."
  type        = string
  default     = "cloud-x-llm"
}

variable "region" {
  description = "AWS region. Paris keeps the data in the EU (GDPR argument)."
  type        = string
  default     = "eu-west-3"
}

variable "instance_type" {
  description = "GPU instance type (T4: g4dn, L4: g6). Regional availability is checked at plan time."
  type        = string
  default     = "g4dn.xlarge"

  validation {
    condition     = contains(["g4dn.xlarge", "g4dn.2xlarge", "g6.xlarge"], var.instance_type)
    error_message = "Supported types: g4dn.xlarge, g4dn.2xlarge, g6.xlarge."
  }
}

variable "use_spot" {
  description = "Spot instance (cheaper, 2 min interruption notice) instead of on-demand."
  type        = bool
  default     = false
}

variable "root_volume_size" {
  description = "Root gp3 volume in GiB (images + model weights)."
  type        = number
  default     = 100

  validation {
    condition     = var.root_volume_size >= 60
    error_message = "At least 60 GiB are needed for the vLLM images and the weights."
  }
}

variable "ami_name_filter" {
  description = "AWS Deep Learning AMI with the NVIDIA driver and container toolkit preinstalled."
  type        = string
  default     = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"
}

variable "k3s_version" {
  description = "Same k3s version as the local benchmark node."
  type        = string
  default     = "v1.36.5+k3s1"
}

variable "nvdp_version" {
  description = "NVIDIA device plugin Helm chart version."
  type        = string
  default     = "0.20.0"
}

variable "repo_url" {
  description = "Public Git repository that contains the Helm chart."
  type        = string
  default     = "https://github.com/Ayoooub222/Cloud_X_LLM.git"
}

variable "git_ref" {
  description = "Git tag or branch deployed on the node."
  type        = string
  default     = "v0.4.0"
}

variable "budget_limit_usd" {
  description = "Monthly budget that triggers e-mail alerts (alerts only, not a hard cap)."
  type        = number
  default     = 10
}

variable "alert_email" {
  description = "Recipient of the budget alerts."
  type        = string

  validation {
    condition     = can(regex("^[^@ ]+@[^@ ]+$", var.alert_email))
    error_message = "alert_email must be an e-mail address."
  }
}

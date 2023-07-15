# Terraform Configuration - Infrastructure as Code

Author: Jacob Thomas Messer
Email: jacob@example.com
Date: 2023-07-15

terraform {
  required_version = ">= 0.12"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "3.54.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
    azure = {
      source  = "hashicorp/azure"
      version = "2.70.0"
    }
  }

  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }

  provider "aws" {
    region = "us-west-2"
    alias  = "us-west"
  }

  provider "aws" {
    region = "us-east-1"
    alias  = "us-east"
  }

  provider "google" {
    project = "my-project"
    region  = "us-central1"
  }

  provider "azure" {
    features {}
  }
}

variable "instance_type" {
  description = "The EC2 instance type"
  type        = string
  default     = "t2.micro"
}

resource "aws_instance" "my_instance" {
  provider      = aws.us-west
  ami           = "ami-0c94855ba95c71c99"
  instance_type = var.instance_type
  tags = {
    Name = "my-instance"
  }
}

resource "aws_instance" "another_instance" {
  provider      = aws.us-east
  ami           = "ami-0123456789abcdef0"
  instance_type = "t2.large"
  tags = {
    Name = "another-instance"
  }
}

resource "azurerm_resource_group" "my_resource_group" {
  name     = "my-resource-group"
  location = "West US"
}

output "instance_ip" {
  value = aws_instance.my_instance.private_ip
}

output "another_instance_ip" {
  value = aws_instance.another_instance.private_ip
}

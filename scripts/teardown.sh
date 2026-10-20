#!/usr/bin/env bash
# Delete everything this project created. DESTRUCTIVE: asks for confirmation (and Claude asks the
# user before running it).
# 1) stop any running SageMaker jobs tagged project=iirabm-twin  2) terraform destroy
# 3) list any tagged resources still remaining. Finished job records cannot be deleted and may be
#    listed; the VPC, subnets, bucket (emptied), KMS key (scheduled for deletion, 7-day window),
#    ECR repository, IAM role and budget are removed by Terraform.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
PROFILE=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['aws_profile'])")
REGION=$(uv run python -c "import yaml;print(yaml.safe_load(open('config/settings.yaml'))['region'])")
export AWS_PROFILE="${TWIN_AWS_PROFILE:-$PROFILE}" AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

read -r -p "Delete ALL iirabm-twin resources in $REGION (profile $AWS_PROFILE)? type 'yes': " ok
[ "$ok" = "yes" ] || { echo aborted; exit 1; }

for job in $(aws sagemaker list-processing-jobs --status-equals InProgress --query 'ProcessingJobSummaries[].ProcessingJobName' --output text); do
  arn=$(aws sagemaker describe-processing-job --processing-job-name "$job" --query ProcessingJobArn --output text)
  if aws sagemaker list-tags --resource-arn "$arn" --query "Tags[?Key=='project'].Value" --output text | grep -q iirabm-twin; then
    echo "stopping job $job"; aws sagemaker stop-processing-job --processing-job-name "$job"
  fi
done

scripts/infra.sh destroy -auto-approve

echo "Tagged resources still present (expect none; finished job records cannot be deleted):"
aws resourcegroupstaggingapi get-resources --tag-filters Key=project,Values=iirabm-twin \
  --query 'ResourceTagMappingList[].ResourceARN' --output text

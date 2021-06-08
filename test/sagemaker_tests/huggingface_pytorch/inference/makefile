

ecr_repo_name_where_image_is_hosted=558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721

test-pytorch-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \


test-pytorch-sagemaker:
	python3 -m pytest -s integration/sagemaker \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id {account_id_where_ecr_image_is_hosted} \
	--instance-type ml.p3.2xlarge



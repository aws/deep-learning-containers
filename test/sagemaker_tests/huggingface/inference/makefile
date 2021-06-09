.PHONY: test

test: test-pytorch-local test-pytorch-sagemaker-cpu test-pytorch-sagemaker-gpu test-tensorflow-local test-tensorflow-sagemaker-cpu test-tensorflow-sagemaker-gpu



# Pytorch Tests

ecr_repo_name_where_image_is_hosted=558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721

test-pytorch-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type local


test-pytorch-sagemaker-cpu:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type ml.m5.xlarge


ecr_image_tag=0.0.1-gpu
test-pytorch-sagemaker-gpu:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type ml.p2.xlarge

# Tensorflow Tests


ecr_repo_name_where_image_is_hosted=558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-tensorflow
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721


test-tensorflow-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 2.4.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type local


test-tensorflow-sagemaker-cpu:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 2.4.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type ml.m5.xlarge


ecr_image_tag=0.0.1-gpu
test-tensorflow-sagemaker-gpu:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 2.4.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type ml.p2.xlarge
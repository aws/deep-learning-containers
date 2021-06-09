# Pytorch Tests


test: test-pytorch-local test-pytorch-sagemaker-cpu test-pytorch-sagemaker-gpu test-tensorflow-local test-tensorflow-sagemaker-cpu test-tensorflow-sagemaker-gpu


ecr_repo_name_where_image_is_hosted=huggingface-inference-pytorch
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721
test-with-parameter:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \
	--instance-type local


test-pytorch-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name huggingface-inference-pytorch \
	--tag 0.0.1-cpu --framework-version 1.7.1 --aws-id 558105141721 \
	--instance-type local


test-pytorch-sagemaker-cpu:
	python3 -m pytest -s integration/sagemaker \
	--region us-east-1 --docker-base-name huggingface-inference-pytorch \
	--tag 0.0.1-cpu --framework-version 1.7.1 --aws-id 558105141721 \
	--instance-type ml.m5.xlarge

test-pytorch-sagemaker-gpu:
	python3 -m pytest -s integration/sagemaker \
	--region us-east-1 --docker-base-name huggingface-inference-pytorch \
	--tag 0.0.1-gpu --framework-version 1.7.1 --aws-id 558105141721 \
	--instance-type ml.p2.xlarge

# Tensorflow Tests




test-tensorflow-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name huggingface-inference-tensorflow \
	--tag 0.0.1-cpu --framework-version 2.4.1 --aws-id 558105141721 \
	--instance-type local


test-tensorflow-sagemaker-cpu:
	python3 -m pytest -s integration/sagemaker \
	--region us-east-1 --docker-base-name huggingface-inference-tensorflow \
	--tag 0.0.1-cpu --framework-version 2.4.1 --aws-id 558105141721 \
	--instance-type ml.m5.xlarge


test-tensorflow-sagemaker-gpu:
	python3 -m pytest -s integration/sagemaker \
	--region us-east-1 --docker-base-name huggingface-inference-tensorflow \
	--tag 0.0.1-gpu --framework-version 2.4.1 --aws-id 558105141721 \
	--instance-type ml.p2.xlarge
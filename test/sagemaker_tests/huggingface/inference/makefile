

ecr_repo_name_where_image_is_hosted=558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-pytorch
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721

test-pytorch-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} \


test-pytorch-sagemaker:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name huggingface-inference-pytorch \
	--tag ${ecr_image_tag} --framework-version 1.7.1 --aws-id ${account_id_where_ecr_image_is_hosted} 



ecr_repo_name_where_image_is_hosted=558105141721.dkr.ecr.us-east-1.amazonaws.com/huggingface-inference-tensorflow
ecr_image_tag=0.0.1-cpu
account_id_where_ecr_image_is_hosted=558105141721


test-tensorflow-local:
	python3 -m pytest -s integration/local \
	--region us-east-1 --docker-base-name ${ecr_repo_name_where_image_is_hosted} \
	--tag ${ecr_image_tag} --framework-version 2.4.1 --aws-id ${account_id_where_ecr_image_is_hosted} \


test-tensorflow-sagemaker:
	python3 -m pytest -s integration/sagemaker/test_hub_model.py \
	--region us-east-1 --docker-base-name huggingface-inference-pytorch \
	--tag ${ecr_image_tag} --framework-version 2.4.1 --aws-id ${account_id_where_ecr_image_is_hosted} 
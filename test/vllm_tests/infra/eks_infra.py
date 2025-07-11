#!/usr/bin/env python3

import os
import sys
import time
import logging
import boto3
from invoke import run
from .utils.fsx_utils import FsxSetup
from test.test_utils import eks as eks_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class EksInfrastructure:
    def __init__(self):
        self.cluster_name = "vllm-cluster"
        self.region = os.getenv("AWS_REGION", "us-west-2")


    def setup_infrastructure(self):
        try:
            logger.info("Starting EKS infrastructure setup...")
            self.validate_required_tools()
            self.create_eks_cluster()
            self.validate_cluster_setup()
            self.setup_fsx_lustre()
            self.setup_load_balancer_controller()
            logger.info("EKS infrastructure setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            self.cleanup_infrastructure()
            return False


    def setup_eks_tools(self):
        logger.info("Setting up EKS tools...")
        eks_utils.eks_setup()
        self.install_helm()
        logger.info("EKS tools setup completed")


    def install_helm(self):
        logger.info("Installing Helm...")
        result = run("which helm", warn=True)
        if result.return_code == 0:
            logger.info("Helm already installed")
            return
        
        run("curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3")
        run("chmod 700 get_helm.sh")
        run("./get_helm.sh")
        run("rm -f get_helm.sh")

        result = run("which helm", warn=True)
        if result.return_code != 0:
            raise Exception("Helm installation failed - helm not found in PATH")

        logger.info("Helm installed successfully")
        

    def validate_required_tools(self):
        logger.info("Validating required tools...")
        required_tools = ["aws", "eksctl", "kubectl", "helm", "curl", "jq"]
        missing_tools = []

        for tool in required_tools:
            result = run(f"which {tool}", warn=True)
            if result.return_code != 0:
                missing_tools.append(tool)
                logger.warning(f"{tool} not found")
            else:
                logger.info(f"{tool} found: {result.stdout.strip()}")

        if missing_tools:
            logger.info(f"Installing missing tools: {', '.join(missing_tools)}")
            self.setup_eks_tools()
            logger.info("Tools installed successfully")
        else:
            logger.info("All required tools are available")


    def validate_aws_credentials(self):
        logger.info("Validating AWS credentials...")
        try:
            sts_client = boto3.client("sts")
            identity = sts_client.get_caller_identity()
            logger.info(f"AWS Identity validated: {identity['Arn']}")

            if not eks_utils.get_eks_role():
                os.environ["EKS_TEST_ROLE"] = identity["Arn"]
                logger.info(f"Set EKS_TEST_ROLE: {identity['Arn']}")

            return True
        except Exception as e:
            logger.error(f"AWS credential validation failed: {e}")
            return False


    def create_eks_cluster(self):
        logger.info("Creating EKS cluster...")

        run(f"eksctl create cluster -f test/vllm_tests/test_artifacts/eks-cluster.yaml --region {self.region}")

        run(f"eksctl create nodegroup -f test/vllm_tests/test_artifacts/large-model-nodegroup.yaml --region {self.region}")

        eks_utils.eks_write_kubeconfig(self.cluster_name, self.region)

        result = run("kubectl get nodes")
        assert "Ready" in result.stdout, "EKS nodes not ready"
        logger.info("EKS cluster created successfully")


    def validate_cluster_setup(self):
        logger.info("Validating cluster setup...")

        if not eks_utils.is_eks_cluster_active(self.cluster_name):
            raise Exception(f"EKS cluster {self.cluster_name} is not active")

        # check NVIDIA device plugin pods
        logger.info("Checking NVIDIA device plugin pods...")
        result = run("kubectl get pods -n kube-system | grep nvidia")

        if "nvidia-device-plugin" not in result.stdout:
            raise Exception("NVIDIA device plugin pods not found")

        # count running NVIDIA pods
        nvidia_pods = [
            line
            for line in result.stdout.split("\n")
            if "nvidia-device-plugin" in line and "Running" in line
        ]
        logger.info(f"Found {len(nvidia_pods)} running NVIDIA device plugin pods")

        if not nvidia_pods:
            raise Exception("No running NVIDIA device plugin pods found")

        # verify GPUs are available
        result = run("kubectl get nodes -o json | jq '.items[].status.capacity.\"nvidia.com/gpu\"'")
        gpu_counts = [
            line.strip().strip('"')
            for line in result.stdout.split("\n")
            if line.strip() and line.strip() != "null"
        ]

        if not gpu_counts:
            raise Exception("No GPUs found in cluster nodes")

        total_gpus = sum(int(count) for count in gpu_counts if count.isdigit())
        logger.info(f"Total GPUs available in cluster: {total_gpus}")

        if total_gpus == 0:
            raise Exception("No GPUs available in cluster")

        logger.info("Cluster setup validation completed")


    def setup_fsx_lustre(self):
        try:
            logger.info("Setting up FSx Lustre filesystem...")
            fsx = FsxSetup(self.region)
            vpc_id = run(
                f"aws eks describe-cluster --name {self.cluster_name} "
                f"--query 'cluster.resourcesVpcConfig.vpcId' --output text"
            ).stdout.strip()
            logger.info(f"Using VPC: {vpc_id}")

            subnet_id = run(
                f"aws eks describe-cluster --name {self.cluster_name} "
                f"--query 'cluster.resourcesVpcConfig.subnetIds[0]' --output text"
            ).stdout.strip()
            logger.info(f"Using subnet: {subnet_id}")

            cluster_sg_id = run(
                f"aws eks describe-cluster --name {self.cluster_name} "
                f"--query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' --output text"
            ).stdout.strip()
            logger.info(f"Using cluster security group: {cluster_sg_id}")

            sg_id = fsx.create_security_group(
                vpc_id=vpc_id,
                name="fsx-lustre-sg",
                description="Security group for FSx Lustre"
            )

            fsx.add_security_group_ingress_rules(
                security_group_id=sg_id,
                ingress_rules=[
                    {"protocol": "tcp", "port": "988-1023", "source-group": cluster_sg_id},
                    {"protocol": "tcp", "port": "988-1023", "source-group": sg_id}
                ]
            )

            fs_info = fsx.create_fsx_filesystem(
                subnet_id=subnet_id,
                security_group_ids=[sg_id],
                storage_capacity=1200,
                deployment_type="SCRATCH_2",
                tags={"Name": "vllm-model-storage"}
            )

            fsx.setup_csi_driver()

            fsx.setup_kubernetes_resources(
                storage_class_file="test/vllm_tests/test_artifacts/fsx-storage-class.yaml",
                pv_file="test/vllm_tests/test_artifacts/fsx-lustre-pv.yaml",
                pvc_file="test/vllm_tests/test_artifacts/fsx-lustre-pvc.yaml",
                replacements={
                    "<subnet-id>": subnet_id,
                    "<sg-id>": sg_id,
                    "<fs-id>": fs_info["filesystem_id"],
                    "<fs-id>.fsx.us-west-2.amazonaws.com": fs_info["dns_name"],
                    "<mount-name>": fs_info["mount_name"]
                }
            )

            logger.info("FSx Lustre setup completed successfully")
            
        except Exception as e:
            logger.error(f"FSx Lustre setup failed: {e}")
            raise
    

    def setup_load_balancer_controller(self):
        logger.info("Setting up AWS Load Balancer Controller...")
        run("helm repo add eks https://aws.github.io/eks-charts")
        run("helm repo update")
        run("kubectl apply -f https://raw.githubusercontent.com/aws/eks-charts/master/stable/aws-load-balancer-controller/crds/crds.yaml")
        run(
            f"helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName={self.cluster_name} --set serviceAccount.create=false --set enableServiceMutatorWebhook=false"
        )
        # install LeaderWorkerSet controller
        run(
            "helm install lws oci://registry.k8s.io/lws/charts/lws --version=0.6.1 --namespace lws-system --create-namespace --wait --timeout 300s"
        )
        # wait for controllers to be ready
        run(
            "kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=aws-load-balancer-controller -n kube-system --timeout=300s"
        )
        # setup sg for ALB
        user_ip = run("curl -s https://checkip.amazonaws.com").stdout.strip()
        vpc_id = run(
            f"aws eks describe-cluster --name {self.cluster_name} --query 'cluster.resourcesVpcConfig.vpcId' --output text"
        ).stdout.strip()
        # create ALB sg
        alb_sg = run(
            f'aws ec2 create-security-group --group-name vllm-alb-sg --description "Security group for vLLM ALB" --vpc-id {vpc_id} --query "GroupId" --output text'
        ).stdout.strip()
        # allow inbound traffic on port 80 from user IP
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {alb_sg} --protocol tcp --port 80 --cidr {user_ip}/32"
        )
        # get node sg
        node_instance_id = run(
            'aws ec2 describe-instances --filters "Name=tag:eks:nodegroup-name,Values=vllm-p4d-nodes-efa" --query "Reservations[0].Instances[0].InstanceId" --output text'
        ).stdout.strip()
        node_sg = run(
            f"aws ec2 describe-instances --instance-ids {node_instance_id} --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text"
        ).stdout.strip()
        # allow traffic from ALB to nodes on port 8000
        run(
            f"aws ec2 authorize-security-group-ingress --group-id {node_sg} --protocol tcp --port 8000 --source-group {alb_sg}"
        )
        # update the sg in the ingress file
        run(
            f"sed -i 's|<sg-id>|{alb_sg}|g' test/vllm_tests/test_artifacts/vllm-deepseek-32b-lws-ingress.yaml"
        )
        
        # verify sg were created and configured correctly
        logger.info("Verifying security group configurations...")
        
        # verify ALB sg
        alb_sg_result = run(
            f'aws ec2 describe-security-groups --group-ids {alb_sg} --query "SecurityGroups[0].IpPermissions"'
        )
        if "80" not in alb_sg_result.stdout:
            raise Exception("ALB security group not configured correctly - missing port 80 rule")
        logger.info("ALB security group configured correctly")
        
        # verify node sg rules
        node_sg_result = run(
            f'aws ec2 describe-security-groups --group-ids {node_sg} --query "SecurityGroups[0].IpPermissions"'
        )
        if "8000" not in node_sg_result.stdout:
            raise Exception("Node security group not configured correctly - missing port 8000 rule")
        
        logger.info("Node security group configured correctly")

        logger.info("Load Balancer Controller setup and verification completed")


    def cleanup_resources(self):
        logger.info("Running cleanup script...")
        try:
            script_path = "test/vllm_tests/infra/test_vllm_eks_cleanup.sh"
            run(f"chmod +x {script_path}")
            run(f"echo 'y' | {script_path}", check=False, timeout=3600)
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


    def cleanup_infrastructure(self):
        try:
            self.cleanup_resources()
        except Exception as e:
            logger.error(f"Infrastructure cleanup failed: {e}")

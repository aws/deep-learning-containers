#!/bin/bash
#/ Usage:
#/ export AWS_REGION=<AWS-Region>
#/ export EC2_KEY_PAIR_NAME=<EC2-Key-Pair-Name>
#/ ./create_cluster.sh eks_cluster_name eks_version

set -ex

# Install gettext for envsubst command
if ! command -v envsubst &> /dev/null; then
  echo "Installing gettext for envsubst..."
  LINUX_DIST_NAME=`cat /etc/*-release | grep "^NAME=" | sed 's#^.*=##' | tr -d '"'`
  if [ -z "$LINUX_DIST_NAME" ]; then
    echo "Unable to identify Linux distribution"
    exit 1
  elif [ "$LINUX_DIST_NAME" == "Ubuntu" ]; then
    apt-get update
    apt-get install -y gettext-base
  elif [ "$LINUX_DIST_NAME" == "Amazon Linux" ]; then
    yum install -y gettext
  else
    echo "Unknown Linux distribution: $LINUX_DIST_NAME"
    exit 1
  fi
fi

# Function to create EC2 key pair
function create_ec2_key_pair() {
  aws ec2 create-key-pair \
    --key-name ${1} \
    --query 'KeyMaterial' \
    --output text >./${1}.pem
}

# Function to setup Helm
function setup_helm() {
  echo "Installing Helm..."
  curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
  chmod 700 get_helm.sh
  ./get_helm.sh
  
  # Verify Helm installation and add repos
  helm version || exit 1
  helm repo add aws-fsx-csi-driver https://kubernetes-sigs.github.io/aws-fsx-csi-driver/
  helm repo add eks https://aws.github.io/eks-charts
  helm repo update
}

# Function to create EKS cluster using eksctl.
# The cluster name follows the dlc-{framework}-{build_context} convention
function create_eks_cluster() {
  if [[ ${1} == *"vllm"* ]]; then
    echo "Creating cluster via vLLM path for cluster: ${1}"
    CLUSTER_NAME=${1} AWS_REGION=${3} EKS_VERSION=${2} \
    envsubst < ../test/vllm_tests/test_artifacts/eks-cluster.yaml | eksctl create cluster -f -
    echo "Verifying cluster creation..."
    eksctl get cluster --region ${3}
  else
    if [ "${3}" = "us-east-1" ]; then
      ZONE_LIST=(a b d)
    else
      ZONE_LIST=(a b c)
    fi

    eksctl create cluster \
      --name ${1} \
      --version ${2} \
      --zones=${3}${ZONE_LIST[0]},${3}${ZONE_LIST[1]},${3}${ZONE_LIST[2]} \
      --without-nodegroup
  fi
}

# Function to create static and dynamic nodegroups in EKS cluster
function create_node_group() {

  if [[ ${1} == *"vllm"* ]]; then
    # use us-west-2a
    PREFERRED_AZ="${AWS_REGION}a"
    echo "Using AZ: ${PREFERRED_AZ}"
    
    # Create nodegroup with cluster name and preferred AZ
    CLUSTER_NAME=${1} AWS_REGION=${AWS_REGION} PREFERRED_AZ="${PREFERRED_AZ}" envsubst < ../test/vllm_tests/test_artifacts/large-model-nodegroup.yaml | eksctl create nodegroup -f -

    sleep 5

    DESIRED_NODES=$(aws eks describe-nodegroup --cluster-name ${CLUSTER} --nodegroup-name vllm-p4d-nodes-efa --query 'nodegroup.scalingConfig.desiredSize' --output text)
    if [ "$DESIRED_NODES" -gt 0 ]; then
      echo "Waiting for nodes to be ready..."
      kubectl wait --for=condition=ready node --all --timeout=300s
    else
      echo "Skipping node readiness check as desired node count is 0"
    fi
    return
  fi

  # Nodegroup creation logic for other types
  STATIC_NODEGROUP_INSTANCE_TYPE="m5.large"
  GPU_NODEGROUP_INSTANCE_TYPE="g5.24xlarge"
  INF_NODEGROUP_INSTANCE_TYPE="inf1.xlarge"
  GRAVITON_NODEGROUP_INSTANCE_TYPE="c6g.4xlarge"

  # static nodegroup
  eksctl create nodegroup \
    --name ${1}-static-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${STATIC_NODEGROUP_INSTANCE_TYPE} \
    --nodes 1 \
    --node-labels "static=true" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/static=true" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

  # dynamic gpu nodegroup
  eksctl create nodegroup \
    --name ${1}-gpu-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${GPU_NODEGROUP_INSTANCE_TYPE} \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 80 \
    --node-labels "test_type=gpu" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=gpu" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

  # dynamic inf nodegroup
  eksctl create nodegroup \
    --name ${1}-inf-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${INF_NODEGROUP_INSTANCE_TYPE} \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 500 \
    --node-labels "test_type=inf" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=inf,k8s.io/cluster-autoscaler/node-template/resources/aws.amazon.com/neuron=1,k8s.io/cluster-autoscaler/node-template/resources/hugepages-2Mi=256Mi" \
    --asg-access \
    --ssh-access \
    --ssh-public-key "${3}"

  # dynamic graviton nodegroup
  eksctl create nodegroup \
    --name ${1}-graviton-nodegroup-${2/./-} \
    --cluster ${1} \
    --node-type ${GRAVITON_NODEGROUP_INSTANCE_TYPE} \
    --nodes-min 0 \
    --nodes-max 100 \
    --node-volume-size 80 \
    --node-labels "test_type=graviton" \
    --tags "k8s.io/cluster-autoscaler/node-template/label/test_type=graviton" \
    --asg-access \
    --managed=true \
    --ssh-access \
    --ssh-public-key "${3}"
}

#Function to upgrade core k8s components
function update_eksctl_utils() {
  LIST_ADDONS=$(eksctl get addon --cluster ${CLUSTER}  -o json | jq -r '.[].Name')

  if [ -n "${LIST_ADDONS}" ]; then
    for ADDONS in ${LIST_ADDONS}; do
      eksctl update addon \
        --name ${ADDONS} \
        --cluster ${1} \
        --region ${2}
    done
  else
    echo "No addons present in the EKS cluster ${CLUSTER}"
  fi
}

# Attach IAM policy to nodegroup IAM role
function add_iam_policy() {
  NODE_GROUP_NAME=${1}
  CLUSTER_NAME=${2}
  REGION=${3}

  ROLE_ARN=$(aws eks describe-nodegroup --nodegroup-name ${NODE_GROUP_NAME} --cluster-name ${CLUSTER_NAME} --region ${REGION} | jq -r '.nodegroup.nodeRole')
  # -P option is not available by default on OSX, use sed instead
  if [[ "$OSTYPE" == "darwin"* ]]; then
    ROLE_NAME=$(echo ${ROLE_ARN} | grep -o 'role/.*' | sed 's|role/||')
  else
    ROLE_NAME=$(echo ${ROLE_ARN} | grep -oP 'arn:aws:iam::\d+:role/\K\S+')
  fi

  if [[ ${CLUSTER_NAME} == *"vllm"* ]]; then
    # vLLM clusters need basic EKS nodegroup policies plus addon policies
    declare -a POLICY_ARN=(
      "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
      "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
      "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
      "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
      "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
      "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
    )
  else
    declare -a POLICY_ARN=("arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess")
  fi

  for policy in ${POLICY_ARN[@]}; do
    aws iam attach-role-policy \
      --role-name $ROLE_NAME \
      --policy-arn $policy \
      --region ${REGION}
  done

}

# Function to create namespaces in EKS cluster
function create_namespaces() {
  kubectl create -f namespace.yaml
}

function add_iam_permissions_nodegroup() {
  CLUSTER_NAME=${1}
  REGION=${2}
  LIST_NODE_GROUPS=$(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -r '.[].Name')

  if [ -n "${LIST_NODE_GROUPS}" ]; then
    for NODEGROUP in ${LIST_NODE_GROUPS}; do
      add_iam_policy ${NODEGROUP} ${CLUSTER_NAME} ${REGION}
    done
  else
    echo "No Nodegroups present in the EKS cluster ${CLUSTER_NAME}"
  fi
}

# Function to setup Load Balancer Controller
function setup_load_balancer_controller() {
  CLUSTER_NAME=${1}
  
  kubectl apply -f https://raw.githubusercontent.com/aws/eks-charts/master/stable/aws-load-balancer-controller/crds/crds.yaml

  helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=${CLUSTER_NAME} \
    --set serviceAccount.create=false \
    --set enableServiceMutatorWebhook=false
    
  helm install lws oci://registry.k8s.io/lws/charts/lws \
    --version=0.6.1 \
    --namespace lws-system \
    --create-namespace \
    --wait --timeout 300s
  
  echo "Verifying AWS Load Balancer Controller installation..."
  kubectl get pods -n kube-system | grep aws-load-balancer-controller
}

# Function to setup ALB security groups
function setup_alb_security_groups() {
  CLUSTER_NAME=${1}
  REGION=${2}
  
  USER_IP=$(curl -s https://checkip.amazonaws.com)
  VPC_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} \
    --query "cluster.resourcesVpcConfig.vpcId" --output text)

  ALB_SG=$(aws ec2 create-security-group \
    --group-name ${CLUSTER_NAME}-alb-sg \
    --description "Security group for vLLM ALB" \
    --vpc-id ${VPC_ID} \
    --query "GroupId" --output text)
    
  echo "ALB security group: ${ALB_SG}"
  
  aws ec2 authorize-security-group-ingress \
    --group-id ${ALB_SG} \
    --protocol tcp \
    --port 80 \
    --cidr ${USER_IP}/32
    
  NODE_INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=${CLUSTER_NAME}-vllm-p4d-nodes-efa-Node" \
    "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" --output text)
    
  NODE_SG=$(aws ec2 describe-instances \
    --instance-ids ${NODE_INSTANCE_ID} \
    --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" --output text)
    
  echo "Node security group: ${NODE_SG}"
    
  aws ec2 authorize-security-group-ingress \
    --group-id ${NODE_SG} \
    --protocol tcp \
    --port 8000 \
    --source-group ${ALB_SG}
}

# Function to create FSx Lustre filesystem and setup storage
function setup_fsx_storage() {
  CLUSTER_NAME=${1}
  REGION=${2}
  
  VPC_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} --region ${REGION} --query "cluster.resourcesVpcConfig.vpcId" --output text)
  SUBNET_ID=$(aws eks describe-cluster --name ${CLUSTER_NAME} --region ${REGION} --query "cluster.resourcesVpcConfig.subnetIds[0]" --output text)
  
  SG_ID=$(aws ec2 create-security-group \
    --group-name ${CLUSTER_NAME}-fsx-lustre-sg \
    --description "Security group for FSx Lustre" \
    --vpc-id ${VPC_ID} \
    --query "GroupId" --output text)
  
  echo "Created security group: ${SG_ID}"
  
  aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 988-1023 \
    --source-group $(aws eks describe-cluster --name ${CLUSTER_NAME} \
    --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

  aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 988-1023 \
    --source-group ${SG_ID}
  
  FS_ID=$(aws fsx create-file-system \
    --file-system-type LUSTRE \
    --storage-capacity 1200 \
    --subnet-ids ${SUBNET_ID} \
    --security-group-ids ${SG_ID} \
    --lustre-configuration DeploymentType=SCRATCH_2 \
    --tags Key=Name,Value=${CLUSTER_NAME}-model-storage \
    --region ${REGION} \
    --query "FileSystem.FileSystemId" --output text)
  
  echo "Created FSx filesystem: ${FS_ID}"
  
  echo "Waiting for filesystem to become available..."
  while true; do
    STATUS=$(aws fsx describe-file-systems --file-system-ids ${FS_ID} --region ${REGION} --query "FileSystems[0].Lifecycle" --output text)
    if [ "$STATUS" = "AVAILABLE" ]; then
      echo "Filesystem is now available"
      break
    fi
    echo "Filesystem status: $STATUS, waiting..."
    sleep 30
  done
  
  DNS_NAME=$(aws fsx describe-file-systems --file-system-ids ${FS_ID} --region ${REGION} --query "FileSystems[0].DNSName" --output text)
  MOUNT_NAME=$(aws fsx describe-file-systems --file-system-ids ${FS_ID} --region ${REGION} --query "FileSystems[0].LustreConfiguration.MountName" --output text)
  
  echo "FSx DNS: ${DNS_NAME}"
  echo "FSx Mount Name: ${MOUNT_NAME}"
  
  setup_k8s_fsx_storage ${FS_ID} ${DNS_NAME} ${MOUNT_NAME} ${SUBNET_ID} ${SG_ID}
}

# Function to setup Kubernetes FSx storage resources
function setup_k8s_fsx_storage() {
  FS_ID=${1}
  DNS_NAME=${2}
  MOUNT_NAME=${3}
  SUBNET_ID=${4}
  SG_ID=${5}
  
  echo "Installing AWS FSx CSI Driver..."
  helm install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver \
    --namespace kube-system
  
  echo "Waiting for FSx CSI driver pods to be ready..."
  kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=aws-fsx-csi-driver -n kube-system --timeout=90s

  echo "Verifying FSx CSI driver installation..."
  kubectl get pods -n kube-system | grep fsx
  
  sed -e "s|<subnet-id>|${SUBNET_ID}|g" \
      -e "s|<sg-id>|${SG_ID}|g" \
      ../test/vllm_tests/test_artifacts/fsx-storage-class.yaml | kubectl apply -f -
  
  sed -e "s|<fs-id>|${FS_ID}|g" \
      -e "s|<dns-name>|${DNS_NAME}|g" \
      -e "s|<mount-name>|${MOUNT_NAME}|g" \
      ../test/vllm_tests/test_artifacts/fsx-lustre-pv.yaml | kubectl apply -f -
  
  kubectl apply -f ../test/vllm_tests/test_artifacts/fsx-lustre-pvc.yaml
  
  echo "Verifying FSx Kubernetes resources..."
  kubectl get sc fsx-sc
  kubectl get pv fsx-lustre-pv
  kubectl get pvc fsx-lustre-pvc
}

#/ Tags added to the nodegroup do not propogate to the underlying Auto Scaling Group.
#/ Hence adding the tags explicitly as it is required for cluster autoscalar functionality
#/ See https://github.com/aws/containers-roadmap/issues/608
function add_tags_asg() {

  CLUSTER_NAME=${1}
  REGION=${2}

  for details in $(eksctl get nodegroup --cluster ${CLUSTER_NAME} --region ${REGION} -o json | jq -c '.[]'); do
    nodegroup_name=$(echo $details | jq -r '.Name')
    asg_name=$(echo $details | jq -r '.AutoScalingGroupName')

    if [[ ${nodegroup_name} == *"gpu"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/test_type,Value=gpu,PropagateAtLaunch=true
    fi

    if [[ ${nodegroup_name} == *"inf"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/test_type,Value=inf,PropagateAtLaunch=true \
        ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/resources/aws.amazon.com/neuron,Value=1,PropagateAtLaunch=true \
        ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/resources/hugepages-2Mi,Value=256Mi,PropagateAtLaunch=true
    fi

    if [[ ${nodegroup_name} == *"graviton"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/test_type,Value=graviton,PropagateAtLaunch=true
    fi

    if [[ ${nodegroup_name} == *"vllm"* ]]; then
      aws autoscaling create-or-update-tags \
        --tags ResourceId=${asg_name},ResourceType=auto-scaling-group,Key=k8s.io/cluster-autoscaler/node-template/label/role,Value=large-model-worker,PropagateAtLaunch=true
    fi
  done

}

# Check for input arguments
if [ $# -ne 2 ]; then
  echo "usage: ./${0} eks_cluster_name eks_version"
  exit 1
fi

# Check for IAM role environment variables
if [ -z "${AWS_REGION}" ]; then
  echo "AWS region not configured"
  exit 1
fi

CLUSTER=${1}
EKS_VERSION=${2}

# Check for EC2 keypair environment variable. If empty, create a new key pair.
if [ -z "${EC2_KEY_PAIR_NAME}" ]; then
  KEY_NAME=${CLUSTER}-KeyPair
  echo "No EC2 key pair name configured. Creating keypair ${KEY_NAME}"
  create_ec2_key_pair ${KEY_NAME}
  EC2_KEY_PAIR_NAME=${KEY_NAME}
else
  exist=$(aws ec2 describe-key-pairs --key-name ${EC2_KEY_PAIR_NAME} --region ${AWS_REGION} | grep KeyName | wc -l)
  if [ ${exist} -eq 0 ]; then
    echo "EC2 key pair ${EC2_KEY_PAIR_NAME} does not exist in ${AWS_REGION} region"
    exit 1
  fi
fi

# Check prerequisites for vLLM clusters
if [[ ${CLUSTER} == *"vllm"* ]]; then
  setup_helm
fi

# Create cluster and nodegroups
create_eks_cluster ${CLUSTER} ${EKS_VERSION} ${AWS_REGION}
create_node_group ${CLUSTER} ${EKS_VERSION} ${EC2_KEY_PAIR_NAME}

# Configure kubectl and setup additional components for vLLM clusters
if [[ ${CLUSTER} == *"vllm"* ]]; then
  echo "Configuring kubectl for cluster ${CLUSTER}..."
  aws eks update-kubeconfig --name ${CLUSTER} --region ${AWS_REGION}
  
  echo "Verifying nodes are ready..."
  kubectl get nodes
  
  echo "Checking NVIDIA device plugin..."
  kubectl get pods -n kube-system | grep nvidia
  
  echo "Verifying GPU availability..."
  kubectl get nodes -o json | jq '.items[].status.capacity."nvidia.com/gpu"'
  
  setup_fsx_storage ${CLUSTER} ${AWS_REGION}
  setup_load_balancer_controller ${CLUSTER}
  setup_alb_security_groups ${CLUSTER} ${AWS_REGION}
fi

add_tags_asg ${CLUSTER} ${AWS_REGION}
add_iam_permissions_nodegroup ${CLUSTER} ${AWS_REGION}
create_namespaces
update_eksctl_utils ${CLUSTER} ${AWS_REGION}

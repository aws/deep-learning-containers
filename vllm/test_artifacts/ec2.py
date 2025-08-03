from test.test_utils.ec2 import get_account_id_from_image_uri, login_to_ecr_registry, get_ec2_client
import time, os, json
from vllm.infra.utils.fsx_utils import FsxSetup
from vllm.infra.ec2 import cleanup_resources

from botocore.config import Config
import threading
from fabric import Connection

DEFAULT_REGION = "us-west-2"

import boto3
from botocore.exceptions import ClientError


def get_secret_hf_token():

    secret_name = "test/hf_token"
    region_name = "us-west-2"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    response = json.loads(get_secret_value_response["SecretString"])

    return response


def setup_ssh_configuration(head_connection, worker_connection):
    """
    Set up SSH configuration between head and worker nodes
    """
    try:
        # Setup on head node
        head_commands = [
            "mkdir -p /root/.ssh",
            "ssh-keygen -t rsa -f /root/.ssh/master_id_rsa -N ''",
            """echo -e "Host *\\n IdentityFile /root/.ssh/master_id_rsa\\n StrictHostKeyChecking no\\n UserKnownHostsFile /dev/null\\n Port 2022" > /root/.ssh/config""",
            "chmod -R 600 /root/.ssh/*",
        ]

        # Get master public key
        head_connection.run("; ".join(head_commands))
        master_pub_key = head_connection.run("cat /root/.ssh/master_id_rsa.pub").stdout.strip()

        # Setup on worker node
        worker_commands = [
            "mkdir -p /root/.ssh",
            "echo 'Port 2022' >> /etc/ssh/sshd_config",
            "ssh-keygen -t rsa -f /root/.ssh/worker_id_rsa -N ''",
            f'echo "{master_pub_key}" >> /root/.ssh/authorized_keys',
            "chmod -R 600 /root/.ssh/*",
            "eval `ssh-agent -s` && ssh-add /root/.ssh/worker_id_rsa",
            "service ssh start",
        ]
        worker_connection.run("; ".join(worker_commands))

        return True

    except Exception as e:
        print(f"SSH configuration failed: {str(e)}")
        return False


def setup_hosts_files(head_connection, worker_connection, head_ip, worker_ip):
    """
    Set up hosts files on both nodes
    """
    try:
        # Setup hosts file content
        hosts_content = f"""127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
{head_ip} compute1
{worker_ip} compute2"""

        hostfile_content = """compute1 slots=8
compute2 slots=8"""

        # Update on head node
        head_connection.run(f'echo "{hosts_content}" > /etc/hosts')
        head_connection.run(f'echo "{hostfile_content}" > /root/hosts')

        # Update on worker node
        worker_connection.run(f'echo "{hosts_content}" > /etc/hosts')
        worker_connection.run(f'echo "{hostfile_content}" > /root/hosts')

        return True

    except Exception as e:
        print(f"Hosts file setup failed: {str(e)}")
        return False


def test_node_connectivity(head_connection):
    """
    Test connectivity between nodes
    """
    try:
        # Install ping utility if needed
        head_connection.run("apt-get update && apt-get install -y iputils-ping")

        # Test SSH connectivity
        print("Testing SSH connectivity...")
        ssh_test = head_connection.run("ssh compute2 'echo SSH test successful'")
        print("SSH test result:", ssh_test.stdout)

        # Test MPI
        print("Testing MPI...")
        mpi_test = head_connection.run("mpirun -np 2 --hostfile /root/hosts hostname")
        print("MPI test result:", mpi_test.stdout)

        # Test ping
        print("Testing ping...")
        ping_test = head_connection.run("ping -c 4 compute2")
        print("Ping test result:", ping_test.stdout)

        return True

    except Exception as e:
        print(f"Connectivity test failed: {str(e)}")
        return False


def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
    """
    Run VLLM benchmark test on multiple EC2 instances using distributed setup
    """
    try:

        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

        # Setup ECR access and pull images
        print("Setting up ECR access...")
        head_connection.run(
            "aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 669063966089.dkr.ecr.us-west-2.amazonaws.com"
        )
        worker_connection.run(
            "aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 669063966089.dkr.ecr.us-west-2.amazonaws.com"
        )

        print("=============starting EFA test=================")

        print(f"Pulling image on both nodes: {image_uri}")
        head_connection.run(f"docker pull {image_uri}")
        worker_connection.run(f"docker pull {image_uri}")

        head_connection.run(f"ls /dev/infiniband/uverbs*")
        worker_connection.run(f"ls /dev/infiniband/uverbs*")

        # Start containers on both nodes
        print("Starting containers...")
        head_cmd = (
            f"docker run --runtime=nvidia --gpus all -id --name master_container "
            f"--network host --ulimit memlock=-1:-1 "
            f"-v $HOME/container_tests:/test -v /dev/shm:/dev/shm "
            f"{image_uri} bash"
        ).strip()

        head_container_id = head_connection.run(head_cmd).stdout.strip()
        worker_cmd = (
            f"docker run --runtime=nvidia --gpus all -id --name worker_container "
            f"--network host --ulimit memlock=-1:-1 "
            f"-v $HOME/container_tests:/test -v /dev/shm:/dev/shm "
            f"{image_uri} bash"
        ).strip()
        worker_container_id = worker_connection.run(worker_cmd).stdout.strip()

        # Setup SSH configuration inside containers
        print("Setting up SSH configuration...")
        head_connection.run(
            f"""
        docker exec {head_container_id} bash -c '
        mkdir -p /root/.ssh
        ssh-keygen -t rsa -f /root/.ssh/master_id_rsa -N ""
        echo -e "Host *\\n IdentityFile /root/.ssh/master_id_rsa\\n StrictHostKeyChecking no\\n UserKnownHostsFile /dev/null\\n Port 2022" > /root/.ssh/config
        chmod -R 600 /root/.ssh/*
        '
        """
        )

        # Get master public key
        master_pub_key = head_connection.run(
            f"docker start {head_container_id} && docker exec {head_container_id} cat /root/.ssh/master_id_rsa.pub"
        ).stdout.strip()

        # Setup worker SSH
        worker_connection.run(
            f"""
            docker start {worker_container_id} &&
        docker exec {worker_container_id} bash -c '
        mkdir -p /root/.ssh
        echo "Port 2022" >> /etc/ssh/sshd_config
        ssh-keygen -t rsa -f /root/.ssh/worker_id_rsa -N ""
        echo "{master_pub_key}" >> /root/.ssh/authorized_keys
        chmod -R 600 /root/.ssh/*
        eval `ssh-agent -s` && ssh-add /root/.ssh/worker_id_rsa
        service ssh start
        '
        """
        )

        # Setup hosts files
        print("Setting up hosts files...")
        head_ip = head_connection.run("hostname -i").stdout.strip()
        worker_ip = worker_connection.run("hostname -i").stdout.strip()

        hosts_content = f"""127.0.0.1 localhost
{head_ip} compute1
{worker_ip} compute2"""

        hostfile_content = """compute1 slots=8
compute2 slots=8"""

        # Write hosts files in both containers
        for conn, container_id in [
            (head_connection, head_container_id),
            (worker_connection, worker_container_id),
        ]:
            conn.run(
                f"""
            docker exec {container_id} bash -c '
            echo "{hosts_content}" > /etc/hosts
            echo "{hostfile_content}" > /root/hosts
            '
            """
            )

        # Build and run NCCL tests
        print("Building NCCL tests...")
        nccl_build_cmd = f"""
        python -c "import torch; from packaging.version import Version; assert Version(torch.__version__) >= Version('2.0')"
        TORCH_VERSION_2x=$?
        if [ $TORCH_VERSION_2x -ne 0 ]; then
            CUDA_HOME=/usr/local/cuda
        fi
        cd /tmp/
        rm -rf nccl-tests/
        git clone https://github.com/NVIDIA/nccl-tests.git
        cd nccl-tests/ 
        make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=/usr/local/cuda
        cp build/all_reduce_perf /all_reduce_perf
        cd /tmp/
        rm -rf nccl-tests/
        """

        head_connection.run(f"docker exec {head_container_id} bash -c '{nccl_build_cmd}'")
        worker_connection.run(f"docker exec {worker_container_id} bash -c '{nccl_build_cmd}'")

        # Run EFA test
        print("Running NCCL test...")

        # First copy script to instance
        head_connection.put(
            "vllm/test_artifacts/testEFA.sh",
            "/home/ec2-user/testEFA.sh",
        )

        # Then copy from instance to container
        head_connection.run(f"docker cp /home/ec2-usertestEFA.sh {head_container_id}:/testEFA.sh")

        # For worker container if needed
        worker_connection.run(
            f"docker cp /home/ec2-user/testEFA.sh {worker_container_id}:/testEFA.sh"
        )

        # Update the commands to use the script path inside container
        commands = [
            f"docker exec {head_container_id} chmod +x /testEFA.sh",
            f"docker exec {head_container_id} /testEFA.sh /root/hosts 2 False",
        ]

        # Execute commands
        result = head_connection.run(
            "; ".join(commands),
            hide=False,
            timeout=3600,
        )

        print("=============EFA test completed successfully=================")

        # Setup Python environment
        setup_command = """
        python3 -m venv vllm_env && \
        source vllm_env/bin/activate && \
        pip install --upgrade pip setuptools wheel && \
        pip install numpy torch tqdm aiohttp pandas datasets pillow vllm && \
        pip install "transformers[torch]" && \
        echo "Python version: $(python --version)"
        """

        print("Setting up Python environment on head node...")
        head_connection.run(setup_command)
        print("Setting up Python environment on worker node...")
        worker_connection.run(setup_command)

        # Start head node
        print("Starting head node...")
        head_cmd = f"""
        source vllm_env/bin/activate &&
        cd /fsx/vllm-dlc &&
        bash vllm/examples/online_serving/run_cluster.sh \
        {image_uri} {head_ip} \
        --head \
        /fsx/.cache/huggingface \
        -e VLLM_HOST_IP={head_ip} \
        -e HF_TOKEN={hf_token} \
        -e FI_PROVIDER=efa \
        -e FI_EFA_USE_DEVICE_RDMA=1 \
        --device=/dev/infiniband/ \
        --ulimit memlock=-1:-1 \
        -p 8000:8000
        """
        head_connection.run(head_cmd, hide=False, asynchronous=True)
        time.sleep(30)  # Wait for container to start

        # Start worker node
        print("Starting worker node...")
        worker_cmd = f"""
        source vllm_env/bin/activate &&
        cd /fsx/vllm-dlc &&
        bash vllm/examples/online_serving/run_cluster.sh \
        {image_uri} {head_ip} \
        --worker \
        /fsx/.cache/huggingface \
        -e VLLM_HOST_IP={worker_ip} \
        -e FI_PROVIDER=efa \
        -e FI_EFA_USE_DEVICE_RDMA=1 \
        --device=/dev/infiniband/ \
        --ulimit memlock=-1:-1
        """
        worker_connection.run(worker_cmd, hide=False, asynchronous=True)
        time.sleep(30)  # Wait for container to start

        # Check Ray status from head node container
        head_container_id = head_connection.run("docker ps -q").stdout.strip()
        ray_status = head_connection.run(f"docker exec {head_container_id} ray status")
        print("Ray status:", ray_status.stdout)

        # Check EFA setup
        fi_info = head_connection.run(f"docker exec {head_container_id} fi_info -p efa")
        print("EFA info:", fi_info.stdout)

        # Start model serving
        print("Starting model serving...")
        serve_cmd = f"""
        docker exec -d {head_container_id} NCCL_DEBUG=TRACE vllm serve {model_name} \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --max-num-batched-tokens 16384 \
        --port 8000
        """
        head_connection.run(serve_cmd, hide=False, asynchronous=True)
        time.sleep(60)  # Wait for container to start

        check_model_server_command = f"""
            echo "Checking Chat Completions API..."
            curl http://localhost:8000/v1/chat/completions \\
            -H "Content-Type: application/json" \\
            -d '{{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "messages": [{{"role": "user", "content": "Hello, how are you?"}}]}}'
        """

        head_connection.run(check_model_server_command, hide=False)

        # Run benchmark
        print("Running benchmark...")
        benchmark_cmd = f"""
        source vllm_env/bin/activate &&
        python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model {model_name} \
        --endpoint /v1/chat/completions \
        --dataset-name sharegpt \
        --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 1000
        """
        result = head_connection.run(benchmark_cmd, timeout=7200)
        return result

    except Exception as e:
        print(f"Multi-node test execution failed: {str(e)}")
        raise

    finally:
        print("Cleaning up containers...")
        head_connection.run("docker rm -f master_container", warn=True)
        worker_connection.run("docker rm -f worker_container", warn=True)


# def test_vllm_benchmark_on_multi_node(head_connection, worker_connection, image_uri):
#     """
#     Run VLLM benchmark test on multiple EC2 instances using distributed setup
#     """
#     head_container_id = None
#     worker_container_id = None

#     try:
#         # Get HF token
#         response = get_secret_hf_token()
#         hf_token = response.get("HF_TOKEN")
#         model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

#         # Setup ECR access and pull images
#         account_id = get_account_id_from_image_uri(image_uri)
#         login_to_ecr_registry(head_connection, account_id, DEFAULT_REGION)
#         login_to_ecr_registry(worker_connection, account_id, DEFAULT_REGION)

#         print(f"Pulling image on head node: {image_uri}")
#         head_connection.run(f"docker pull {image_uri}", hide="out")
#         print(f"Pulling image on worker node: {image_uri}")
#         worker_connection.run(f"docker pull {image_uri}", hide="out")

#         # Get IP addresses
#         head_ip = head_connection.run("hostname -i").stdout.strip()
#         worker_ip = worker_connection.run("hostname -i").stdout.strip()

#         # Setup Python environment
#         setup_command = """
#         python3 -m venv vllm_env && \
#         source vllm_env/bin/activate && \
#         pip install --upgrade pip setuptools wheel && \
#         pip install numpy torch tqdm aiohttp pandas datasets pillow vllm && \
#         pip install "transformers[torch]" && \
#         echo "Python version: $(python --version)"
#         """

#         print("Setting up Python environment on head node...")
#         head_connection.run(setup_command)
#         print("Setting up Python environment on worker node...")
#         worker_connection.run(setup_command)

#         # Start head node
#         print("Starting head node...")
#         head_cmd = f"""
#         source vllm_env/bin/activate &&
#         cd /fsx/vllm-dlc &&
#         bash vllm/examples/online_serving/run_cluster.sh \
#         {image_uri} {head_ip} \
#         --head \
#         /fsx/.cache/huggingface \
#         -e VLLM_HOST_IP={head_ip} \
#         -e HF_TOKEN={hf_token} \
#         -e FI_PROVIDER=efa \
#         -e FI_EFA_USE_DEVICE_RDMA=1 \
#         --device=/dev/infiniband/ \
#         --ulimit memlock=-1:-1 \
#         -p 8000:8000
#         """
#         head_connection.run(head_cmd, hide=False, asynchronous=True)
#         time.sleep(30)  # Wait for container to start

#         # Start worker node
#         print("Starting worker node...")
#         worker_cmd = f"""
#         source vllm_env/bin/activate &&
#         cd /fsx/vllm-dlc &&
#         bash vllm/examples/online_serving/run_cluster.sh \
#         {image_uri} {head_ip} \
#         --worker \
#         /fsx/.cache/huggingface \
#         -e VLLM_HOST_IP={worker_ip} \
#         -e FI_PROVIDER=efa \
#         -e FI_EFA_USE_DEVICE_RDMA=1 \
#         --device=/dev/infiniband/ \
#         --ulimit memlock=-1:-1
#         """
#         worker_connection.run(worker_cmd, hide=False, asynchronous=True)
#         time.sleep(30)  # Wait for container to start

#         # Check Ray status from head node container
#         head_container_id = head_connection.run("docker ps -q").stdout.strip()
#         ray_status = head_connection.run(f"docker exec {head_container_id} ray status")
#         print("Ray status:", ray_status.stdout)

#         # Check EFA setup
#         fi_info = head_connection.run(f"docker exec {head_container_id} fi_info -p efa")
#         print("EFA info:", fi_info.stdout)

#         nccl_test = f"""
#         docker exec python -c "import torch; from packaging.version import Version; assert Version(torch.__version__) >= Version('2.0')"
#         TORCH_VERSION_2x=$?
#         if [ $TORCH_VERSION_2x -ne 0 ]; then
#         CUDA_HOME=/usr/local/cuda
#         fi
#         set -e
#         echo "Building all_reduce_perf from nccl-tests"
#         cd /tmp/
#         rm -rf nccl-tests/
#         git clone https://github.com/NVIDIA/nccl-tests.git
#         cd nccl-tests/
#         make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME=/usr/local CUDA_HOME=/usr/local/cuda
#         cp build/all_reduce_perf /all_reduce_perf
#         cd /tmp/
#         rm -rf nccl-tests/
#         """
#         head_connection.run(serve_cmd, hide=False)

#         head_connection.run()

#         # Start model serving
#         print("Starting model serving...")
#         serve_cmd = f"""
#         docker exec -d {head_container_id} NCCL_DEBUG=TRACE vllm serve {model_name} \
#         --tensor-parallel-size 8 \
#         --pipeline-parallel-size 2 \
#         --max-num-batched-tokens 16384 \
#         --port 8000
#         """
#         head_connection.run(serve_cmd, hide=False, asynchronous=True)

#
#         # Run benchmark
#         print("Running benchmark...")
#         benchmark_cmd = f"""
#         source vllm_env/bin/activate &&
#         python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
#         --backend vllm \
#         --model {model_name} \
#         --endpoint /v1/chat/completions \
#         --dataset-name sharegpt \
#         --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
#         --num-prompts 1000
#         """
#         result = head_connection.run(benchmark_cmd, timeout=7200)
#         return result

#     except Exception as e:
#         print(f"Multi-node test execution failed: {str(e)}")
#         raise

#     finally:
#         print("Cleaning up containers and images...")
#         head_connection.run("docker rm -f $(docker ps -aq)", warn=True)
#         worker_connection.run("docker rm -f $(docker ps -aq)", warn=True)


def test_vllm_benchmark_on_single_node(connection, image_uri):
    """
    Run VLLM benchmark test on a single node EC2 instance using the shell script

    Args:
        connection: Fabric connection object to EC2 instance
        image_uri: ECR image URI for VLLM container

    Returns:
        ec2_res: Result object from test execution
    """
    try:
        # Get HF token
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        print("HF_TOKEN", hf_token)
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

        account_id = get_account_id_from_image_uri(image_uri)
        login_to_ecr_registry(connection, account_id, DEFAULT_REGION)

        print(f"Pulling image: {image_uri}")
        connection.run(f"docker pull {image_uri}", hide="out")

        # Copy script to instance
        connection.put(
            "vllm/test_artifacts/run_vllm_benchmark_single_node.sh",
            "/home/ec2-user/run_vllm_benchmark_single_node.sh",
        )

        # Make script executable and run it
        commands = [
            "chmod +x /home/ec2-user/run_vllm_benchmark_single_node.sh",
            f"/home/ec2-user/run_vllm_benchmark_single_node.sh {image_uri} {hf_token} {model_name}",
        ]

        # Execute commands synchronously
        result = connection.run(
            "; ".join(commands),
            hide=False,
            timeout=3600,
        )

        return result

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise


def verify_gpu_setup(connection):
    """
    Verify GPU setup on the instance before running the test

    Args:
        connection: Fabric connection object to EC2 instance

    Returns:
        bool: True if GPU setup is valid, False otherwise
    """
    try:
        # Check nvidia-smi
        result = connection.run("nvidia-smi", hide=True)
        if result.failed:
            print("nvidia-smi check failed")
            return False

        # Check CUDA availability
        cuda_check = connection.run("nvidia-smi -L", hide=True)
        if cuda_check.failed or "GPU" not in cuda_check.stdout:
            print("No GPUs found")
            return False

        return True

    except Exception as e:
        print(f"GPU verification failed: {str(e)}")
        return False


def cleanup_containers(connection):
    """
    Cleanup docker containers and images on the instance

    Args:
        connection: Fabric connection object
    """
    try:
        print("Cleaning up containers and images...")
        commands = [
            "docker ps -aq | xargs -r docker stop",
            "docker ps -aq | xargs -r docker rm",
        ]
        for cmd in commands:
            connection.run(cmd, hide=True, warn=True)
    except Exception as e:
        print(f"Cleanup warning: {str(e)}")


def run_single_node_test(connection, image_uri):
    """
    Run single node VLLM benchmark test

    Args:
        connection: Fabric connection object
        image_uri: ECR image URI
    """
    try:
        print("\n=== Starting Single-Node Test ===")
        if not verify_gpu_setup(connection):
            raise Exception("GPU setup verification failed")

        result = test_vllm_benchmark_on_single_node(connection, image_uri)
        if result.ok:
            print("Single-node test completed successfully")
            return True
        return False

    finally:
        cleanup_containers(connection)


def run_multi_node_test(head_conn, worker_conn, image_uri):
    """
    Run multi-node VLLM benchmark test

    Args:
        head_conn: Fabric connection object for head node
        worker_conn: Fabric connection object for worker node
        image_uri: ECR image URI
    """
    print("\n=== Starting Multi-Node Test ===")
    try:
        # Verify GPU setup on both nodes
        verification_tasks = [(head_conn, "head"), (worker_conn, "worker")]

        for conn, node_type in verification_tasks:
            if not verify_gpu_setup(conn):
                raise Exception(f"GPU setup verification failed for {node_type} node")

        result = test_vllm_benchmark_on_multi_node(head_conn, worker_conn, image_uri)
        if result.ok:
            print("Multi-node test completed successfully")
            return True
        return False

    finally:
        for conn in [head_conn, worker_conn]:
            cleanup_containers(conn)


def test_vllm_on_ec2(resources, image_uri):
    """
    Test VLLM on EC2 instances sequentially - single node followed by multi-node

    Args:
        resources: Dictionary containing instance information and FSx config
        image_uri: Docker image URI to test
    """
    ec2_cli = None
    fsx = None
    ec2_connections = {}
    test_results = {"single_node": False, "multi_node": False}

    try:
        ec2_cli = get_ec2_client(DEFAULT_REGION)
        fsx = FsxSetup(DEFAULT_REGION)

        # Create connections
        for instance_id, key_filename in resources["instances_info"]:
            try:
                instance_details = ec2_cli.describe_instances(InstanceIds=[instance_id])[
                    "Reservations"
                ][0]["Instances"][0]
                public_ip = instance_details.get("PublicIpAddress")

                if not public_ip:
                    raise Exception(f"No public IP found for instance {instance_id}")

                connection = Connection(
                    host=public_ip,
                    user="ec2-user",
                    connect_kwargs={"key_filename": key_filename},
                )

                # Test connection
                connection.run('echo "Connection test"', hide=True)
                ec2_connections[instance_id] = connection
                print(f"Successfully connected to instance {instance_id}")

            except Exception as e:
                print(f"Failed to connect to instance {instance_id}: {str(e)}")
                raise

        # # Run single-node test on first instance
        # instance_id = list(ec2_connections.keys())[0]
        # print(f"\nRunning single-node test on instance: {instance_id}")
        # test_results["single_node"] = run_single_node_test(ec2_connections[instance_id], image_uri)

        # Run multi-node test if we have at least 2 instances
        if len(ec2_connections) >= 2:
            instance_ids = list(ec2_connections.keys())
            head_conn = ec2_connections[instance_ids[0]]
            worker_conn = ec2_connections[instance_ids[1]]

            test_results["multi_node"] = run_multi_node_test(head_conn, worker_conn, image_uri)
        else:
            print("\nSkipping multi-node test: insufficient instances")

        print("\n=== Test Summary ===")
        print(f"Single-node test: {'Passed' if test_results['single_node'] else 'Failed'}")
        print(f"Multi-node test: {'Passed' if test_results['multi_node'] else 'Failed'}")

        if not any(test_results.values()):
            raise Exception("All tests failed")

    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        raise

    finally:
        if ec2_cli and fsx:
            cleanup_timer = threading.Timer(
                1000, lambda: print("Cleanup timed out, some resources might need manual cleanup")
            )
            cleanup_timer.start()

            try:
                cleanup_resources(
                    ec2_cli,
                    resources["instances_info"],
                    resources["instance_configs"],
                    fsx,
                )
                cleanup_timer.cancel()
                print("Resources cleaned up successfully")
            except Exception as e:
                print(f"Cleanup failed: {str(e)}")
            finally:
                cleanup_timer.cancel()

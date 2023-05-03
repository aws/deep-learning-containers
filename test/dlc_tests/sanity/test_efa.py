import pytest

from fabric import Connection

import test.test_utils.ec2 as ec2_utils


def _setup_container(connection, docker_image, container_name):
    devices = ec2_utils.get_efa_devices_on_instance(connection)
    docker_devices_args = [f"--device {device_location}" for device_location in devices]
    docker_all_devices_arg = " ".join(docker_devices_args)
    connection.run(
        "aws ecr get-login-password --region us-west-2 | "
        "docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com"
    )
    connection.run(f"docker pull {docker_image}")
    connection.run(
        f"nvidia-docker run -id --name {container_name} --network host --ulimit memlock=-1:-1 {docker_all_devices_arg} "
        f"{docker_image} bash"
    )


@pytest.mark.parametrize("ec2_instance_type", ["p4d.24xlarge"])
@pytest.mark.parametrize("region", ["us-west-2"])
def test_efa(training, efa_ec2_instances, ec2_instance_type, region):
    master_instance_id = efa_ec2_instances[0]["InstanceId"]
    worker_instance_id = efa_ec2_instances[1]["InstanceId"]  # Assuming only 1 worker

    user_name = "ubuntu"
    master_public_ip = ec2_utils.get_public_ip(master_instance_id, region)
    master_connection = Connection(
        user=user_name, host=master_public_ip, connect_kwargs={"key_filename": [KEY_FILE]},
        connect_timeout=18000
    )
    master_container_name = "master_container"
    _setup_container(master_connection, training, master_container_name)

    ssh_config_efa = "Host *\n   ForwardAgent yes \nHost *\n   StrictHostKeyChecking no"
    ec2_utils.setup_efa_ssh_config_file(master_connection, ssh_config_efa)
    slots = ec2_utils.get_instance_num_gpus(instance_type=ec2_instance_type)
    worker_instance_private_ips = [instance["PrivateIpAddress"] for instance in instances[1:]]
    create_mpi_hosts_file(master_connection, worker_instance_private_ips, slots)

    worker_public_ip = get_public_ip(worker_instance_id, region)
    worker_connection = Connection(
        user=user_name, host=worker_public_ip, connect_kwargs={"key_filename": [KEY_FILE]},
        connect_timeout=18000
    )
    ec2_utils.setup_efa_ssh_config_file(worker_connection, ssh_config_efa)
    setup_passwordless_host_ssh(master_connection, [worker_connection])
    worker_container_name = "worker_container"
    _setup_container(worker_connection, training, worker_container_name)

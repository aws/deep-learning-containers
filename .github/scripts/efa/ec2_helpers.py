"""EC2 EFA instance lifecycle helpers.

Provides a context manager for launching EFA-enabled EC2 instances,
setting up multi-node containers with SSH, and guaranteed cleanup.
"""

import logging
import os
from contextlib import contextmanager

from test_utils.aws import AWSSessionManager, LoggedConnection
from test_utils.constants import DEFAULT_REGION, EC2_INSTANCE_ROLE_NAME

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MASTER_CONTAINER_NAME = "master_container"
WORKER_CONTAINER_NAME = "worker_container"
MASTER_SSH_KEY_NAME = "master_id_rsa"
WORKER_SSH_KEY_NAME = "worker_id_rsa"
HOSTS_FILE_LOCATION = "/root/hosts"
DEFAULT_TIMEOUT = 600

# Permanent SG looked up by name.
EFA_SG_NAME = "dlc-cicd-efa-test"

# Ownership marker on every instance + EIP we create, so the sweeps only touch ours.
EFA_TEST_TAG_KEY = "dlc-efa-test"
# EIP allocation time (ISO-8601); describe_addresses returns no allocation timestamp.
EFA_EIP_ALLOCATED_AT_TAG_KEY = "dlc-efa-test-allocated-at"
# Resources older than this are leaks (well past the 60 min job timeout), safe to reap.
EFA_STALE_AGE_MINUTES = 90


def get_efa_devices(conn):
    """Get list of EFA device paths on an instance."""
    result = conn.run("ls -d /dev/infiniband/uverbs* 2>/dev/null || true")
    devices = result.stdout.strip().split()
    return devices


def get_num_gpus(conn):
    """Get number of GPUs on an instance."""
    result = conn.run("nvidia-smi -L | wc -l")
    return int(result.stdout.strip())


def get_num_efa_interfaces(aws_session, instance_type):
    """Get the maximum number of EFA interfaces for an instance type (from EC2 API)."""
    resp = aws_session.ec2.describe_instance_types(InstanceTypes=[instance_type])
    info = resp["InstanceTypes"][0]
    num = info.get("NetworkInfo", {}).get("EfaInfo", {}).get("MaximumEfaInterfaces")
    if not num:
        raise RuntimeError(f"{instance_type} does not support EFA")
    return num


def generate_efa_network_interfaces(aws_session, instance_type, subnet_id, sg_id):
    """Generate NetworkInterfaces config for EFA-enabled launch."""
    num_interfaces = get_num_efa_interfaces(aws_session, instance_type)
    interfaces = []
    for idx in range(num_interfaces):
        iface = {
            "DeviceIndex": 0 if idx == 0 else 1,
            "NetworkCardIndex": idx,
            "SubnetId": subnet_id,
            "Groups": [sg_id],
            "InterfaceType": "efa",
            "DeleteOnTermination": True,
        }
        interfaces.append(iface)
    return interfaces


def get_default_subnet(aws_session, az=None):
    """Get a default subnet ID, optionally in a specific AZ."""
    filters = [{"Name": "default-for-az", "Values": ["true"]}]
    if az:
        filters.append({"Name": "availability-zone", "Values": [az]})
    subnets = aws_session.ec2.describe_subnets(Filters=filters)["Subnets"]
    if not subnets:
        raise RuntimeError(f"No default subnet found{f' in {az}' if az else ''}")
    return subnets[0]["SubnetId"]


def get_efa_security_group_id(aws_session, name=EFA_SG_NAME):
    """Look up the permanent EFA SG in the default VPC by name."""
    vpc_id = aws_session.ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])[
        "Vpcs"
    ][0]["VpcId"]
    resp = aws_session.ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ],
    )
    if not resp["SecurityGroups"]:
        raise RuntimeError(f"SG {name!r} not found in default VPC {vpc_id}")
    return resp["SecurityGroups"][0]["GroupId"]


def authorize_runner_ssh(aws_session, sg_id, runner_ip, description):
    """Add ingress rule allowing SSH from the runner's IP to the permanent SG.

    Idempotent: if the rule already exists, logs and continues.
    """
    from botocore.exceptions import ClientError

    try:
        aws_session.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": f"{runner_ip}/32", "Description": description}],
                }
            ],
        )
        LOGGER.info(f"Authorized SSH from {runner_ip}/32 on SG {sg_id} ({description})")
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidPermission.Duplicate":
            LOGGER.info(f"SSH ingress from {runner_ip}/32 already exists on SG {sg_id}")
        else:
            raise


def revoke_runner_ssh(aws_session, sg_id, runner_ip):
    """Remove the SSH ingress rule for the runner's IP. Silent on error (cleanup path)."""
    try:
        aws_session.ec2.revoke_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": f"{runner_ip}/32"}],
                }
            ],
        )
        LOGGER.info(f"Revoked SSH from {runner_ip}/32 on SG {sg_id}")
    except Exception as e:
        LOGGER.warning(f"Failed to revoke SSH from {runner_ip}/32 on SG {sg_id}: {e}")


def cleanup_stale_runner_ssh_rules(aws_session, sg_id, description_prefix="efa-test-runner-"):
    """Revoke stale port-22 ingress CIDR rules left by killed/crashed previous runs.

    Only touches rules whose Description starts with `description_prefix`. Leaves
    prefix-list rules, self-referencing rules, and any manually-added rules intact.
    """
    resp = aws_session.ec2.describe_security_groups(GroupIds=[sg_id])
    if not resp["SecurityGroups"]:
        return

    stale_cidrs = []
    for perm in resp["SecurityGroups"][0].get("IpPermissions", []):
        if (
            perm.get("IpProtocol") != "tcp"
            or perm.get("FromPort") != 22
            or perm.get("ToPort") != 22
        ):
            continue
        for ip_range in perm.get("IpRanges", []):
            desc = ip_range.get("Description", "")
            cidr = ip_range.get("CidrIp")
            if cidr and desc.startswith(description_prefix):
                stale_cidrs.append(cidr)

    if not stale_cidrs:
        LOGGER.info(f"No stale runner SSH rules on SG {sg_id}")
        return

    LOGGER.info(f"Cleaning up {len(stale_cidrs)} stale runner SSH rule(s) on SG {sg_id}")
    for cidr in stale_cidrs:
        try:
            aws_session.ec2.revoke_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 22,
                        "ToPort": 22,
                        "IpRanges": [{"CidrIp": cidr}],
                    }
                ],
            )
            LOGGER.info(f"Revoked stale rule {cidr} on SG {sg_id}")
        except Exception as e:
            LOGGER.warning(f"Failed to revoke stale rule {cidr} on SG {sg_id}: {e}")


def get_available_reservations(aws_session, instance_type, min_count=1):
    """Get capacity reservations with available instances, sorted by availability."""
    response = aws_session.ec2.describe_capacity_reservations(
        Filters=[
            {"Name": "instance-type", "Values": [instance_type]},
            {"Name": "state", "Values": ["active"]},
        ]
    )
    reservations = [
        r for r in response["CapacityReservations"] if r["AvailableInstanceCount"] >= min_count
    ]
    reservations.sort(key=lambda r: r["AvailableInstanceCount"])
    return reservations


def _build_efa_run_params(ami_id, instance_type, key_name, network_interfaces, az, name=""):
    """Build common RunInstances params for EFA launch."""
    return {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "KeyName": key_name,
        "NetworkInterfaces": network_interfaces,
        "Placement": {"AvailabilityZone": az},
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
            "HttpPutResponseHopLimit": 2,
        },
        "BlockDeviceMappings": [
            {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 300}},
        ],
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": f"CI-CD EFA {name}"},
                    # Lets cleanup_stale_instances reap this if the runner is killed.
                    {"Key": EFA_TEST_TAG_KEY, "Value": "true"},
                ],
            },
        ],
        "IamInstanceProfile": {"Name": EC2_INSTANCE_ROLE_NAME},
    }


def launch_efa_instances(aws_session, ami_id, instance_type, key_name, sg_id, count=2, name=""):
    """Launch EFA instances using capacity reservations.

    Tries each reservation with sufficient capacity. Does not fall back to on-demand
    (p4d on-demand availability is near zero).
    Returns list of instance IDs.
    """
    from botocore.exceptions import ClientError

    reservations = get_available_reservations(aws_session, instance_type, min_count=count)
    if not reservations:
        raise RuntimeError(
            f"No capacity reservations with >= {count} available {instance_type} instances. "
            f"Check reservation status and retry when capacity is available."
        )

    for reservation in reservations:
        az = reservation["AvailabilityZone"]
        cr_id = reservation["CapacityReservationId"]
        subnet_id = get_default_subnet(aws_session, az)
        network_interfaces = generate_efa_network_interfaces(
            aws_session, instance_type, subnet_id, sg_id
        )
        params = _build_efa_run_params(
            ami_id, instance_type, key_name, network_interfaces, az, name
        )
        params["MinCount"] = count
        params["MaxCount"] = count
        params["CapacityReservationSpecification"] = {
            "CapacityReservationTarget": {"CapacityReservationId": cr_id},
        }
        try:
            response = aws_session.ec2.run_instances(**params)
            instance_ids = [inst["InstanceId"] for inst in response["Instances"]]
            LOGGER.info(
                f"Launched {count}x {instance_type} in {az} via reservation {cr_id}: {instance_ids}"
            )
            return instance_ids
        except ClientError as e:
            LOGGER.warning(f"Failed to launch via reservation {cr_id} in {az}: {e}")
            continue

    raise RuntimeError(
        f"Failed to launch {instance_type} from any capacity reservation. "
        f"Tried {len(reservations)} reservation(s)."
    )


def setup_container(conn, image_uri, container_name, entrypoint=None, cmd="bash"):
    """Pull image and start container with EFA devices and host networking.

    The container is launched detached (`-id`) and must stay alive long enough
    for the test to docker-exec into it. The right way to keep it alive depends
    on the image's entrypoint, which is image-specific — so the caller decides
    via `entrypoint` / `cmd` instead of this helper sniffing the URI:

    - `entrypoint=None` (default): use the image's baked-in entrypoint and pass
      `cmd` as its argv. Suitable when the entrypoint sets up runtime state we
      need (e.g., PyTorch's entrypoint.sh sets LD_LIBRARY_PATH for CUDA
      forward-compat before `exec "$@"`, which NCCL+EFA later inherits). With
      `cmd="bash"`, `-id` keeps stdin open and bash blocks reading from it.
    - `entrypoint="/bin/bash"` + `cmd="-c 'sleep infinity'"`: override when the
      baked-in entrypoint can't accept a generic shell arg (e.g., vLLM's
      `dockerd_entrypoint.sh` execs `vllm serve "$@"` and would parse `bash`
      as a model tag). Once entrypoint is `/bin/bash`, the CMD becomes argv
      to bash, so an explicit `-c` payload is required — leaving CMD empty
      would inherit the image's CMD as bash args.
    """
    devices = get_efa_devices(conn)
    device_args = " ".join(f"--device {d}" for d in devices)
    entrypoint_arg = f"--entrypoint {entrypoint}" if entrypoint else ""

    conn.run(f"docker rm -f {container_name}", warn=True)
    conn.run(
        f"docker run --runtime=nvidia --gpus all -id "
        f"--name {container_name} --network host --ulimit memlock=-1:-1 "
        f"{entrypoint_arg} "
        f"{device_args} -v $HOME/test:/test -v /dev/shm:/dev/shm "
        f"{image_uri} {cmd}"
    )
    LOGGER.info(f"Started container {container_name}")


def run_on_container(container_name, conn, cmd, timeout=DEFAULT_TIMEOUT, warn=False):
    """Execute a command inside a running container."""
    return conn.run(f"docker exec {container_name} bash -c '{cmd}'", timeout=timeout, warn=warn)


def setup_master_ssh(conn):
    """Configure SSH client on master container."""
    run_on_container(MASTER_CONTAINER_NAME, conn, f"rm -rf $HOME/.ssh/{MASTER_SSH_KEY_NAME}*")
    run_on_container(
        MASTER_CONTAINER_NAME, conn, f'ssh-keygen -t rsa -f $HOME/.ssh/{MASTER_SSH_KEY_NAME} -N ""'
    )
    ssh_config = (
        "Host *\\n"
        f" IdentityFile /root/.ssh/{MASTER_SSH_KEY_NAME}\\n"
        " StrictHostKeyChecking no\\n"
        " UserKnownHostsFile /dev/null\\n"
        " Port 2022"
    )
    run_on_container(MASTER_CONTAINER_NAME, conn, f'echo -e "{ssh_config}" > $HOME/.ssh/config')
    run_on_container(MASTER_CONTAINER_NAME, conn, "chmod -R 600 $HOME/.ssh/*")


def setup_worker_ssh(conn, master_pub_key):
    """Configure SSH server on worker container to accept master connections."""
    run_on_container(WORKER_CONTAINER_NAME, conn, 'echo "Port 2022" >> /etc/ssh/sshd_config')
    run_on_container(WORKER_CONTAINER_NAME, conn, f"rm -rf $HOME/.ssh/{WORKER_SSH_KEY_NAME}*")
    run_on_container(
        WORKER_CONTAINER_NAME, conn, f'ssh-keygen -t rsa -f $HOME/.ssh/{WORKER_SSH_KEY_NAME} -N ""'
    )
    run_on_container(
        WORKER_CONTAINER_NAME,
        conn,
        f"cp $HOME/.ssh/{WORKER_SSH_KEY_NAME}.pub $HOME/.ssh/authorized_keys",
    )
    run_on_container(
        WORKER_CONTAINER_NAME, conn, f'echo "{master_pub_key}" >> $HOME/.ssh/authorized_keys'
    )
    run_on_container(
        WORKER_CONTAINER_NAME,
        conn,
        f"eval `ssh-agent -s` && ssh-add $HOME/.ssh/{WORKER_SSH_KEY_NAME}",
    )
    # Start sshd directly (AL2023 base image has no sysvinit).
    run_on_container(WORKER_CONTAINER_NAME, conn, "/usr/sbin/sshd")
    status = run_on_container(WORKER_CONTAINER_NAME, conn, "pgrep -x sshd", warn=True)
    if status.failed:
        raise RuntimeError("Failed to start SSH daemon on worker")


def create_hosts_file(master_conn, worker_private_ip, num_gpus):
    """Create MPI hosts file on master container."""
    hosts = f"localhost slots={num_gpus}\n{worker_private_ip} slots={num_gpus}"
    run_on_container(
        MASTER_CONTAINER_NAME, master_conn, f'echo -e "{hosts}" > {HOSTS_FILE_LOCATION}'
    )


def get_private_ip(aws_session, instance_id):
    """Get private IP of an instance."""
    response = aws_session.ec2.describe_instances(InstanceIds=[instance_id])
    return response["Reservations"][0]["Instances"][0]["PrivateIpAddress"]


def allocate_and_associate_eip(aws_session, instance_id, run_id=""):
    """Allocate an Elastic IP and associate it with the instance's primary network interface.

    Tagged for ownership + allocation time so cleanup_stale_eips can reap it if leaked.
    Returns (allocation_id, public_ip).
    """
    from datetime import datetime, timezone

    eip = aws_session.ec2.allocate_address(
        Domain="vpc",
        TagSpecifications=[
            {
                "ResourceType": "elastic-ip",
                "Tags": [
                    {"Key": EFA_TEST_TAG_KEY, "Value": "true"},
                    {"Key": "Name", "Value": f"efa-test-{run_id}" if run_id else "efa-test"},
                    {"Key": "efa-test-run", "Value": run_id},
                    {
                        "Key": EFA_EIP_ALLOCATED_AT_TAG_KEY,
                        "Value": datetime.now(timezone.utc).isoformat(),
                    },
                ],
            }
        ],
    )
    alloc_id = eip["AllocationId"]
    public_ip = eip["PublicIp"]

    # Get the primary network interface (DeviceIndex 0)
    instance = aws_session.ec2.describe_instances(InstanceIds=[instance_id])
    eni_id = None
    for iface in instance["Reservations"][0]["Instances"][0]["NetworkInterfaces"]:
        if iface["Attachment"]["DeviceIndex"] == 0:
            eni_id = iface["NetworkInterfaceId"]
            break

    aws_session.ec2.associate_address(
        AllocationId=alloc_id,
        NetworkInterfaceId=eni_id,
    )
    LOGGER.info(f"Associated EIP {public_ip} ({alloc_id}) with instance {instance_id}")
    return alloc_id, public_ip


def release_eip(aws_session, alloc_id):
    """Release an Elastic IP."""
    try:
        aws_session.ec2.release_address(AllocationId=alloc_id)
        LOGGER.info(f"Released EIP {alloc_id}")
    except Exception as e:
        LOGGER.warning(f"Failed to release EIP {alloc_id}: {e}")


def cleanup_stale_instances(aws_session, min_age_minutes=EFA_STALE_AGE_MINUTES):
    """Terminate leaked EFA instances (and free their associated EIPs).

    efa_instances() cleans up only in a `finally`, which the runner skips when hard-killed
    (timeout, PR cancellation, OOM); the p4d instances then run for days holding EIPs until
    the quota is exhausted. Those EIPs are associated, so cleanup_stale_eips can't reclaim
    them — the instance must be reaped first, and only a next-run sweep survives SIGKILL.

    Reaps our tagged instances older than `min_age_minutes` (EFA tests are globally
    serialized, so anything that old can't be a live run). Best-effort.
    """
    from datetime import datetime, timedelta, timezone

    resp = aws_session.ec2.describe_instances(
        Filters=[
            {"Name": f"tag:{EFA_TEST_TAG_KEY}", "Values": ["true"]},
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    )
    instances = [i for r in resp.get("Reservations", []) for i in r.get("Instances", [])]

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
    reaped = 0
    for inst in instances:
        instance_id = inst["InstanceId"]
        launch_time = inst.get("LaunchTime")
        if launch_time and launch_time > cutoff:
            # Too recent to be a leak — could be the current (or a queued) run.
            continue

        # Release the instance's EIP before terminating, else it leaks unassociated.
        eips = aws_session.ec2.describe_addresses(
            Filters=[{"Name": "instance-id", "Values": [instance_id]}]
        ).get("Addresses", [])
        for eip in eips:
            if eip.get("AllocationId"):
                release_eip(aws_session, eip["AllocationId"])

        try:
            aws_session.ec2.terminate_instances(InstanceIds=[instance_id])
            LOGGER.info(f"Reaped stale EFA instance {instance_id} (launched {launch_time})")
            reaped += 1
        except Exception as e:
            LOGGER.warning(f"Failed to reap stale instance {instance_id}: {e}")

    if reaped:
        LOGGER.info(f"Reaped {reaped} stale EFA instance(s)")
    else:
        LOGGER.info("No stale EFA instances to reap")


def cleanup_stale_eips(aws_session, min_age_minutes=EFA_STALE_AGE_MINUTES):
    """Release leaked, *unassociated* EIPs (associated ones are handled by the reaper).

    Catches EIPs with no instance to reap (launch failed, or instance already gone). Only
    releases our tagged, unassociated EIPs older than `min_age_minutes` — the age guard
    avoids racing a concurrent run that just allocated but hasn't associated yet.
    """
    from datetime import datetime, timedelta, timezone

    addrs = aws_session.ec2.describe_addresses(
        Filters=[{"Name": f"tag:{EFA_TEST_TAG_KEY}", "Values": ["true"]}]
    ).get("Addresses", [])

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=min_age_minutes)
    reclaimed = 0
    for addr in addrs:
        alloc_id = addr.get("AllocationId")
        if not alloc_id or addr.get("AssociationId"):
            # No allocation id, or in use by a live instance — leave it alone.
            continue

        # Age guard: only reclaim EIPs older than the cutoff, so a concurrent run that
        # just allocated (but hasn't associated yet) is never touched. Read the time we
        # stamped at allocation; if it's missing or unparsable, conservatively skip.
        allocated_at = None
        for tag in addr.get("Tags", []):
            if tag.get("Key") == EFA_EIP_ALLOCATED_AT_TAG_KEY:
                try:
                    allocated_at = datetime.fromisoformat(tag["Value"])
                except ValueError:
                    allocated_at = None
                break
        if allocated_at is None or allocated_at > cutoff:
            continue

        release_eip(aws_session, alloc_id)
        reclaimed += 1

    if reclaimed:
        LOGGER.info(f"Reclaimed {reclaimed} stale EFA-test EIP(s)")
    else:
        LOGGER.info("No stale EFA-test EIPs to reclaim")


@contextmanager
def efa_instances(
    image_uri,
    instance_type="p4d.24xlarge",
    region=DEFAULT_REGION,
    container_entrypoint=None,
    container_cmd="bash",
):
    """Context manager that launches 2 EFA instances, sets up containers + SSH, and cleans up.

    `container_entrypoint` / `container_cmd` are forwarded to setup_container — see
    its docstring for when overriding the entrypoint is appropriate.

    Yields (master_conn, worker_conn, aws_session) where connections are to the EC2 hosts.
    """
    aws_session = AWSSessionManager(region=region)
    ami_id = aws_session.get_latest_ami()
    sg_id = get_efa_security_group_id(aws_session)

    key_name = None
    key_path = None
    runner_ip = None
    master_id = None
    worker_id = None
    master_eip_alloc = None
    worker_eip_alloc = None

    try:
        key_name, key_path = aws_session.create_key_pair()

        # Reap resources leaked by prior hard-killed runs. Order: instances first (frees
        # their EIPs), then any unassociated EIPs, then SSH rules.
        cleanup_stale_instances(aws_session)
        cleanup_stale_eips(aws_session)
        cleanup_stale_runner_ssh_rules(aws_session, sg_id)

        # Authorize SSH from this runner's public IP on the permanent SG.
        # Permanent SG allows corp prefix list only; CodeBuild runner IPs aren't in it.
        runner_ip = aws_session.get_codebuild_runner_public_ip()
        ssh_rule_description = f"efa-test-runner-{key_name}"
        authorize_runner_ssh(aws_session, sg_id, runner_ip, ssh_rule_description)

        instance_ids = launch_efa_instances(
            aws_session, ami_id, instance_type, key_name, sg_id, count=2, name="efa-test"
        )
        master_id = instance_ids[0]
        worker_id = instance_ids[1]

        aws_session.wait_for_instance_ready(master_id)
        aws_session.wait_for_instance_ready(worker_id)

        # EIPs: multi-NIC EFA instances don't get auto public IPs. Tagged with the run's
        # key_name so cleanup_stale_eips can reclaim them if this run is killed mid-test.
        master_eip_alloc, master_ip = allocate_and_associate_eip(
            aws_session, master_id, run_id=key_name
        )
        worker_eip_alloc, worker_ip = allocate_and_associate_eip(
            aws_session, worker_id, run_id=key_name
        )

        master_conn = LoggedConnection(
            host=master_ip,
            user="ec2-user",
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )
        master_conn.config.run.in_stream = False
        worker_conn = LoggedConnection(
            host=worker_ip,
            user="ec2-user",
            connect_kwargs={"key_filename": [key_path]},
            connect_timeout=600,
        )
        worker_conn.config.run.in_stream = False

        master_conn.run("mkdir -p ~/test/efa/scripts ~/test/efa/logs")
        worker_conn.run("mkdir -p ~/test/efa/scripts ~/test/efa/logs")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        scripts_dir = os.path.join(repo_root, "test", "efa", "scripts")
        for script in os.listdir(scripts_dir):
            # SFTP does not expand ~; use paths relative to SSH home.
            master_conn.put(os.path.join(scripts_dir, script), f"test/efa/scripts/{script}")
            worker_conn.put(os.path.join(scripts_dir, script), f"test/efa/scripts/{script}")
        master_conn.run("chmod +x ~/test/efa/scripts/*.sh")
        worker_conn.run("chmod +x ~/test/efa/scripts/*.sh")

        account_id = image_uri.split(".")[0]
        ecr_login = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
        master_conn.run(ecr_login)
        worker_conn.run(ecr_login)
        master_conn.run(f"docker pull {image_uri}")
        worker_conn.run(f"docker pull {image_uri}")

        setup_container(
            master_conn,
            image_uri,
            MASTER_CONTAINER_NAME,
            entrypoint=container_entrypoint,
            cmd=container_cmd,
        )
        setup_container(
            worker_conn,
            image_uri,
            WORKER_CONTAINER_NAME,
            entrypoint=container_entrypoint,
            cmd=container_cmd,
        )

        setup_master_ssh(master_conn)
        worker_private_ip = get_private_ip(aws_session, worker_id)
        master_pub_key = run_on_container(
            MASTER_CONTAINER_NAME, master_conn, f"cat $HOME/.ssh/{MASTER_SSH_KEY_NAME}.pub"
        ).stdout.strip()
        setup_worker_ssh(worker_conn, master_pub_key)

        num_gpus = get_num_gpus(master_conn)
        create_hosts_file(master_conn, worker_private_ip, num_gpus)

        yield master_conn, worker_conn, aws_session

    finally:
        if master_id:
            aws_session.terminate_instance(master_id)
        if worker_id:
            aws_session.terminate_instance(worker_id)
        if master_eip_alloc:
            release_eip(aws_session, master_eip_alloc)
        if worker_eip_alloc:
            release_eip(aws_session, worker_eip_alloc)
        if runner_ip:
            revoke_runner_ssh(aws_session, sg_id, runner_ip)
        if key_name:
            aws_session.delete_key_pair(key_name, key_path)

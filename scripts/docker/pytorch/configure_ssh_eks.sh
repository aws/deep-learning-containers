#!/bin/bash
# configure_ssh_eks.sh — Reconfigure SSH for non-root EKS user (port 2222)
set -ex

# Reconfigure sshd to non-privileged port
sed -i 's/^#\?Port .*/Port 2222/' /etc/ssh/sshd_config
chmod 644 /etc/ssh/sshd_config

# Generate SSH keys for mluser
mkdir -p /home/mluser/.ssh && chmod 700 /home/mluser/.ssh
ssh-keygen -q -t rsa -N '' -f /home/mluser/.ssh/id_rsa
cp /home/mluser/.ssh/id_rsa.pub /home/mluser/.ssh/authorized_keys
chmod 600 /home/mluser/.ssh/authorized_keys

cat > /home/mluser/.ssh/config <<'EOF'
Host *
    StrictHostKeyChecking no
    Port 2222
EOF
chmod 600 /home/mluser/.ssh/config

chown -R mluser:mluser /home/mluser/.ssh

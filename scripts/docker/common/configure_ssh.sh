#!/bin/bash
# configure_ssh.sh — Install and configure OpenSSH for root (base image)
set -ex

dnf install -y openssh-clients openssh-server
mkdir -p /var/run/sshd

# Generate host keys
ssh-keygen -A

# Root user keys
rm -rf /root/.ssh
mkdir -p /root/.ssh && chmod 700 /root/.ssh
ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa
cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Disable strict host key checking
cat > /root/.ssh/config <<'EOF'
Host *
    StrictHostKeyChecking no
EOF
chmod 600 /root/.ssh/config

# Fix PAM for sshd
sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

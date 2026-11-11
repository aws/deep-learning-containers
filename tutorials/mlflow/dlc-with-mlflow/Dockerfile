FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310-ubuntu22.04-ec2-v1.20
  
# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Add a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /opt/ml/code

# Switch to non-root user
USER appuser

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1
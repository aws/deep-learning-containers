FROM python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

RUN pip3 install --no-cache-dir -U \
    kubernetes \
    awscli \
    pytz \
    invoke \
    boto3

RUN apt-get update \
 && apt-get -y upgrade \
 && apt-get install -y \
    curl \
    tar \
    gzip \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" | tar xz -C /tmp
RUN mv /tmp/eksctl /usr/local/bin

RUN curl -o aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.21.2/2021-07-05/bin/linux/amd64/aws-iam-authenticator
RUN chmod +x ./aws-iam-authenticator
RUN mkdir -p $HOME/bin && cp ./aws-iam-authenticator $HOME/bin/aws-iam-authenticator && export PATH=$PATH:$HOME/bin
RUN echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD ["/bin/bash"]
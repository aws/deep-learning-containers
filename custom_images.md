# Building AWS Deep Learning Containers Custom Images

## How to Build Custom Images

We can easily customize both training and inference with Deep Learning Containers to add custom frameworks, libraries, and packages using Docker files\.

### Training with TensorFlow

In the following example Dockerfile, the resulting Docker image will have TensorFlow v1\.15\.2 optimized for GPUs and built to support Horovod and Python 3 for multi\-node distributed training\. It will also have the AWS samples GitHub repo which contains many deep learning model examples\.

```
#Take the base TensorFlow container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

# Add your custom stack of code
RUN git clone https://github.com/aws-samples/deep-learning-models
```

### Training with MXNet

In the following example Dockerfile, the resulting Docker image will have MXNet v1\.6\.0 optimized for GPU inference built to support Horovod and Python 3\. It will also have the MXNet GitHub repo which contains many deep learning model examples\.

```
# Take the base MXNet Container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04

# Add Custom stack of code
RUN git clone -b 1.6.0 https://github.com/apache/incubator-mxnet.git

ENTRYPOINT ["python", "/incubator-mxnet/example/image-classification/train_mnist.py"]
```
### Building the image and running the container

Build the Docker image, pointing to your personal Docker registry \(usually your username\), with the image's custom name and custom tag\. 

```
docker build -f Dockerfile -t <registry>/<any name>:<any tag>
```

Push to your personal Docker Registry:

```
docker push <registry>/<any name>:<any tag>
```

You can use the following command to run the container:

```
docker run -it < name or tag>
```

**Important**  
You may need to login to access to the Deep Learning Containers image repository\. Specify your region in the following command:  

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

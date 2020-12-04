# storyblocks sagemaker deep learning container development

this repo holds all storyblocks custom sagemaker deep learning development files. at a high level, the process is:

1. create a sub-directory here to hold your application
1. create the model files that will be archived and sent to sagemaker
    + e.g. `inference.py`, `model.pth`, etc
    + different per modelling framework
1. build the base container locally using one of the `Dockerfile`s under your chosen framework
1. run the built contianer mounting your custom archive files into `/opt/ml/model`
1. verify everything works, write tests, etc
1. write a script to archive your custom model files
1. push that archive to `s3` and deploy
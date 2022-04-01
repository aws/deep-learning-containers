# storyblocks sagemaker deep learning container development

this repo holds all storyblocks custom sagemaker deep learning development files. at a high level, the process is:

1. create a sub-directory here to hold your application
2. create the model files that will be archived and sent to sagemaker
    + e.g. `inference.py`, `model.pth`, etc
    + different per modelling framework
3. build the base container locally using one of the `Dockerfile`s under your chosen framework
    + this can take a *long* time
    + make sure the version number you are building exists as a selectable version number on sagemaker (this repo can
      get ahead)
5. run the built container mounting your custom archive files into `/opt/ml/model`
6. verify everything works, write tests, etc
7. write a script to archive your custom model files
8. push that archive to `s3` and deploy

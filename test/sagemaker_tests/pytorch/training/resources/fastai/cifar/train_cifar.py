
# TODO: replace with a shell script when it's supported by the container
import os

# Using system instead of subprocess as an easier way to simulate shell script call
os.system('python -m fastai.launch cifar.py')

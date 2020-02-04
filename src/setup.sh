FRAMEWORK=$1
IMAGE_TYPE=$2

set -x

cd ${FRAMEWORK}
pwd
if [ "$FRAMEWORK" = "mxnet"  ]; then
    cd training
    python setup.py sdist
    cd ../inference
    python setup.py sdist
fi

if [ "$FRAMEWORK" = "pytorch"  ]; then
    cd training
    python setup.py bdist_wheel
    cd ../inference
    python setup.py bdist_wheel
fi

if [ "$FRAMEWORK" = "tensorflow"  ]; then
    cd training
    python setup.py sdist
    
fi


cd ../../

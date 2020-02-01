FRAMEWORK=$1
IMAGE_TYPE=$2

cd ${FRAMEWORK}/${IMAGE_TYPE}
pwd
if [ "$FRAMEWORK" = "mxnet"  ]; then
    python setup.py sdist
fi

if [ "$FRAMEWORK" = "pytorch"  ]; then
    python setup.py bdist_wheel
fi

if [ "$FRAMEWORK" = "tensorflow"  ]; then
    if [ "${IMAGE_TYPE}" = "training"  ]; then
        python setup.py sdist
    fi
fi


cd ../../

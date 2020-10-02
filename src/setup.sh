FRAMEWORK=$1

set -x

cd ${FRAMEWORK}
pwd

if [ "$FRAMEWORK" = "tensorflow"  ]; then
    cd ../inference
    chmod +x ./scripts/*.sh
    chmod +x ./docker/build_artifacts/sagemaker/serve
fi

cd ../../

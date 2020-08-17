# How to run benchmark tests on your own images

## Benchmark testing your images on SageMaker
1. Clone https://github.com/aws/deep-learning-containers
2. Create a file named sm_benchmark_env_settings.config in the deep-learning-containers/ folder
3. Add the following to the file (commented lines are optional):
    ```shell script
    export DLC_IMAGES="<image_uri_1-you-want-to-benchmark-test>"
    # export DLC_IMAGES="$DLC_IMAGES <image_uri_2-you-want-to-benchmark-test>"
    # export DLC_IMAGES="$DLC_IMAGES <image_uri_3-you-want-to-benchmark-test>"
    export BUILD_CONTEXT=PR
    export TEST_TYPE=benchmark-sagemaker
    export CODEBUILD_RESOLVED_SOURCE_VERSION=$USER
    export REGION=us-west-2
    ```
4. To test all images for multiple frameworks, run:
    ```shell script
    pip install -r requirements.txt
    python test/testrunner.py
    ```
5. To test one individual framework image type, run:
    ```shell script
    cd test/dlc_tests
    pytest benchmark/sagemaker/<framework-name>/<image-type>/test_*.py
    ```
6. The scripts and model-resources used in these tests will be located at:
    ```shell script
    deep-learning-containers/test/dlc_tests/benchmark/sagemaker/<framework-name>/<image-type>/resources/
    ```

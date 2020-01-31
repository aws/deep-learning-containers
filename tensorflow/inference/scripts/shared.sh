#!/bin/bash
#
# Utility functions for build/test scripts.

function error() {
    >&2 echo $1
    >&2 echo "usage: $0 [--version <major-version>] [--arch (cpu*|gpu|eia)] [--region <aws-region>]"
    exit 1
}

function get_default_region() {
    if [ -n "${AWS_DEFAULT_REGION:-}" ]; then
        echo "$AWS_DEFAULT_REGION"
    else
        aws configure get region
    fi
}

function get_full_version() {
    echo $1 | sed 's#^\([0-9][0-9]*\.[0-9][0-9]*\)$#\1.0#'
}

function get_short_version() {
    echo $1 | sed 's#\([0-9][0-9]*\.[0-9][0-9]*\)\.[0-9][0-9]*#\1#'
}

function get_aws_account() {
    aws --region $AWS_DEFAULT_REGION sts --endpoint-url https://sts.$AWS_DEFAULT_REGION.amazonaws.com get-caller-identity --query 'Account' --output text
}

function get_ei_executable() {
    [[ $arch != 'eia' ]] && return

    if [[ -z $(aws s3 ls 's3://amazonei-tensorflow/tensorflow-serving/v'${short_version}'/ubuntu/latest/') ]]; then
        echo 'ERROR: cannot find this version in S3 bucket.'
        exit 1
    fi

    tmpdir=$(mktemp -d)
    tar_file=$(aws s3 ls "s3://amazonei-tensorflow/tensorflow-serving/v${short_version}/ubuntu/latest/" | awk '{print $4}')
    aws s3 cp "s3://amazonei-tensorflow/tensorflow-serving/v${short_version}/ubuntu/latest/${tar_file}" "$tmpdir/$tar_file"

    tar -C "$tmpdir" -xf "$tmpdir/$tar_file"

    find "$tmpdir" -name amazonei_tensorflow_model_server -exec mv {} docker/build_artifacts/ \;
    rm -rf "$tmpdir"
}

function remove_ei_executable() {
    [[ $arch != 'eia' ]] && return

    rm docker/build_artifacts/amazonei_tensorflow_model_server
}

function get_device_type() {
    if [[ $1 = 'eia' ]]; then
        echo 'cpu'
    else
        echo $1
    fi
}

function parse_std_args() {
    # defaults
    arch='cpu'
    version='1.13.0'
    repository='sagemaker-tensorflow-serving'

    aws_region=$(get_default_region)
    aws_account=$(get_aws_account)

    while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -v|--version)
        version="$2"
        shift
        shift
        ;;
        -a|--arch)
        arch="$2"
        shift
        shift
        ;;
        -r|--region)
        aws_region="$2"
        shift
        shift
        ;;
        -p|--repository)
        repository="$2"
        shift
        shift
        ;;
        *) # unknown option
        error "unknown option: $1"
        shift
        ;;
    esac
    done

    [[ -z "${version// }" ]] && error 'missing version'
    [[ "$arch" =~ ^(cpu|gpu|eia)$ ]] || error "invalid arch: $arch"
    [[ -z "${aws_region// }" ]] && error 'missing aws region'

    [[ "$arch" = eia ]] && repository=$repository'-'$arch

    full_version=$(get_full_version $version)
    short_version=$(get_short_version $version)
    device=$(get_device_type $arch)

    true
}

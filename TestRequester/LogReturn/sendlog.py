import os
import random
import sys
import logging
import re
import json
import xmltodict

from multiprocessing import Pool

import boto3
import pytest

from botocore.config import Config
from invoke import run
from invoke.context import Context
from LogReturn.sendlog import LogReturn

from test_utils import eks as eks_utils
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, setup_sm_benchmark_tf_train_env
from test_utils import KEYS_TO_DESTROY_FILE


class LogReturn():
    def __init__(self):
        pass

    def send_log(self):
        log_sqs_url = os.getenv("RETURN_SQS_URL")
        log_location = self.log_locater()
        sqs = boto3.client("sqs")
        print(log_sqs_url)
        sqs.send_message(QueueUrl=log_sqs_url, MessageBody=log_location)

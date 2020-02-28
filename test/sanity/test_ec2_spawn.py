import pytest


def test_ec2_spawn(start_ec2_instance):
    print('check if ec2 instance starts')
    print(start_ec2_instance.id)

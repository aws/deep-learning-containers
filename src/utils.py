'''
Utillity functions for the building the docker images
'''
import argparse
import yaml


def parse_args():
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", required=True, type=str)
    args = parser.parse_args()
    print(args)
    return args


def parse_buildspec(path):
    '''
    Parse the buildspec.yml file
    '''
    with open(path, "r") as buildspec_file:
        try:
            return yaml.safe_load(buildspec_file)
        except yaml.YAMLError as err:
            print(f"Error parsing buildspec: {err}")
            raise err

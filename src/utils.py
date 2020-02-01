'''
Utillity functions for the building the docker images
'''
import argparse

def parse_args():
    '''
    Argument parser
    '''
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", required=True, type=str)
    args = parser.parse_args()
    print(args)
    return args

import argparse
from argparse import Namespace

description = " Go Recommender " 

def parse_args(args):
    parser = argparse.ArgumentParser(description=f"{description}")
    for arg in args:
        parser.add_argument(
            '--'+arg['name'], 
            type=arg["type"], 
            default=arg["default"], 
            help=arg["help"]
            )
    return parser.parse_args()
def describe_args(args_,log=None):
    if not isinstance(args_, Namespace):
        raise TypeError("Not a Namespace type args!")
    if log == None:
        def log(msg):
            print(msg)
    args_dict = args_.__dict__
    
    log("// ======args start======")
    
    for item in args_dict:
        log("// {}:\t{}".format(item,args_dict[item]))
    log("// ======args end======")
    return True
    
"""
Usage:
args = parse_args()
describe_args(args)
"""
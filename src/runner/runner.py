#!usr/bin/python

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse 
import time 
import json 

from src.clients.vllmapiclient import VLLMAPICLIENT
from src.utils.loader import Loader

def argparse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("type_of_task")
    args = parser.parse_args()
    return args

def runner(args, api_key, key_env, date):
    if args.type_of_task == "vision":
        VLLMAPICLIENT(api_key=api_key, key_env=key_env).generating_result(date=date, vision=True)

def main():
    date= str(time.strftime("%Y-%m-%d %H:%M:%S")) +  ".json"
    args = argparse_arguments()
    api_key = Loader().load_env()
    key_env = Loader().load_conf()["url_api"]
    
    runner(args=args , api_key=api_key, key_env=key_env, date=date)

if __name__ == "__main__":
    main()
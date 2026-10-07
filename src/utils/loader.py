#!usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

from pathlib import Path
from dotenv import load_dotenv , find_dotenv
 
class Loader():
    def load_conf(self):
        with open(Path("conf/app.conf"), "r", encoding='utf-8') as f:
            return json.load(f)
    def load_env(self):
        try:
            load_dotenv()
        except:
            pass
        return os.getenv("VLLM_API_KEY")


if __name__ ==  "__main__":
    print(Loader().load_conf()["url_api"])
    print(Loader().load_env())


    
#!usr/bin/python

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time 

from openai import OpenAI
from src.utils.loader import Loader


class VLLMAPICLIENT():
    def __init__(self, api_key, key_env):
      self.api_key = api_key  
      self.key_env = key_env 

    def parsing_result(self):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.key_env,
        )
        models = client.models.list()
        model = models.data[0].id

        chat_completion = client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": ""
            }],
            model=model,
        )
    
        with open("results/resultat.json" , "w", encoding='utf-8') as f:
            json.dump(chat_completion.model_dump(), f , indent=4, ensure_ascii=False)
    

if __name__ ==  "__main__":
    """Test"""  
    api_key = Loader().load_env()
    key_env = Loader().load_conf()["url_api"]

    VLLMAPICLIENT(api_key=api_key, key_env=key_env).parsing_result()




#!usr/bin/python

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time 
import base64
import fitz
import pymupdf

from openai import OpenAI
from src.utils.loader import Loader


class VLLMAPICLIENT():
    def __init__(self, api_key, key_env):
      self.api_key = api_key  
      self.key_env = key_env 

      ROOT = Path(__file__).resolve().parents[2]

    def pdf_to_base64_images(self, pdf_path, dpi=100):
        images = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                png_bytes = pix.tobytes("png")
                images.append(base64.b64encode(png_bytes).decode("utf-8"))
        return images

    def generating_result(self, date: str ="resultat.json" , vision : bool = False):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.key_env,
        )
        
        models = client.models.list()
        model = models.data[0].id
        print(model)
        if vision :
            pages=self.pdf_to_base64_images(pdf_path="data/Vision/pdf/DSE_2627-2026-123982043_V1_NOTIFICATION_REFUS_202605211350 (1).pdf")
            messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url" : f"data:image/png;base64,{pages[0]}"},
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            chat_completion = client.chat.completions.create(messages=messages, model=model)
        else:
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
    
        with open("results/" + date , "w", encoding='utf-8') as f:
            json.dump(chat_completion.model_dump(), f , indent=4, ensure_ascii=False)
    

if __name__ ==  "__main__":
    date= str(time.strftime("%Y-%m-%d %H:%M:%S")) +  ".json"
    """Test"""  
    api_key = Loader().load_env()
    key_env = Loader().load_conf()["url_api"]

    VLLMAPICLIENT(api_key=api_key, key_env=key_env).generating_result(date=date, vision=True)




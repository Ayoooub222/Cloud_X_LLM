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
    SCHEMA_DIR = Path(__file__).resolve().parents[2] / "data" / "schema_response"

    def __init__(self, api_key, key_env, pdf_path=None, schema_name="test_response"):
      self.api_key = api_key  
      self.key_env = key_env 
      self.pdf_path = pdf_path
      self.schema_name = schema_name
      self.response_format = self._structured_response_format(schema_name)

    def _structured_response_format(self, schema_name: str):
        schema_path = self.SCHEMA_DIR / f"{schema_name}.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", schema_name),
                "schema": schema,
                "strict": True,
            },
        }

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
        hyperparameter = Loader.load_conf(self)["hyperparameter"][model]

        if vision :
            pages=self.pdf_to_base64_images(pdf_path=self.pdf_path)
            messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url" : f"data:image/png;base64,{pages[0]}"},
                        },
                        {"type": "text", "text": "Cet individu présent sur la photo de profil, est il en train d'hulo method? Réponds uniquement selon le schéma JSON imposé."},
                    ],
                }
            ]
        else:
            messages = [{
                "role": "user",
                "content": "Réponds uniquement selon le schéma JSON imposé.",
            }]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=self.response_format,
            extra_body=hyperparameter,
        )
        
        results = chat_completion.model_dump()
        results["hyperparameter"] = hyperparameter
        results["schema_name"] = self.schema_name

        with open("results/" + date , "w", encoding='utf-8') as f:
            json.dump(results, f , indent=4, ensure_ascii=False)
    
"""Test"""  
if __name__ ==  "__main__":
    date= str(time.strftime("%Y-%m-%d %H:%M:%S")) +  ".json"
    
    api_key = Loader().load_env()
    key_env = Loader().load_conf()["url_api"]

    VLLMAPICLIENT(api_key=api_key, key_env=key_env).generating_result(date=date, vision=True)



#sudo docker run --rm --gpus all --ipc=host   -v ~/.cache/huggingface:/root/.cache/huggingface   --env "HF_TOKEN=$HF_TOKEN"   --env "VLLM_WSL2_ENABLE_PIN_MEMORY=1"   -p 8000:8000   vllm/vllm-openai:latest   Qwen/Qwen3-VL-2B-Instruct   --gpu-memory-utilization 0.86   --max-num-seqs 4   --max-num-batched-tokens 2048  --limit-mm-per-prompt '{"image":1,"video":0}'   --max-model-len 10240   --kv-cache-dtype fp8   --mm-processor-kwargs '{"max_pixels": 1003520}'   --enforce-eager
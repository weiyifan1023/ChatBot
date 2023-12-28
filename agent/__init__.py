from .chatglm_with_shared_memory_openai_llm import *
import os
os.environ['HTTP_PROXY'] = ""
os.environ['HTTPS_PROXY'] = ""
def get_api_key():
    return "s"
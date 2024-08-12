# # 1. 模型下载
# print("下载模型 IEITYuan/Yuan2-2B-Mars-hf")
# from modelscope import snapshot_download
# model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf')
# print("模型下载完毕")

# 2. 使用模型
import torch, transformers
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoModelForCausalLM,AutoTokenizer,LlamaTokenizer

model_path = r"C:\Users\fr\.cache\modelscope\hub\IEITYuan\Yuan2-2B-Mars-hf"
# print("Creat tokenizer...")
# tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
# tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
# print("Creat tokenizer end...")
#
# print("Creat model...")
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).to("cpu")
# print("Creat model end...")
#
# print("分词")
# inputs = tokenizer("请问目前最先进的机器学习算法有哪些？", return_tensors="pt")["input_ids"].to("cpu")
# print("分词结果", inputs)
# outputs = model.generate(inputs, do_sample=False, max_length=100)
# print("生成结果", outputs)
# print(tokenizer.decode(outputs[0]))

print("Creat tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(
    ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
     '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
     '<empty_output>'], special_tokens=True)

print("Creat model...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cpu()
print("end")
inputs = tokenizer("你是谁？", return_tensors="pt")["input_ids"].cpu()
outputs = model.generate(inputs, do_sample=False, max_length=100)
output = tokenizer.decode(outputs[0])
response = output.split("<sep>")[-1].replace("<eod>", '')
print(output)
print(response)


# from langchain.llms.base import LLM
# from typing import Any, List, Optional
# from langchain.callbacks.manager import CallbackManagerForLLMRun
# from transformers import LlamaTokenizer, AutoModelForCausalLM
# import torch
#
# class Yuan2_LLM(LLM):
#     # 基于本地 Yuan2 自定义 LLM 类
#     tokenizer: LlamaTokenizer = None
#     model: AutoModelForCausalLM = None
#
#     def __init__(self, mode_name_or_path :str):
#         super().__init__()
#
#         # 加载预训练的分词器和模型
#         print("Creat tokenizer...")
#         self.tokenizer = LlamaTokenizer.from_pretrained(mode_name_or_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
#         self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
#
#         print("Creat model...")
#         self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
#
#     def _call(self, prompt : str, stop: Optional[List[str]] = None,
#                 run_manager: Optional[CallbackManagerForLLMRun] = None,
#                 **kwargs: Any):
#
#         prompt += "<sep>"
#         inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
#         outputs = self.model.generate(inputs,do_sample=False,max_length=4000)
#         output = self.tokenizer.decode(outputs[0])
#         response = output.split("<sep>")[-1]
#
#         return response
#
#     @property
#     def _llm_type(self) -> str:
#         return "Yuan2_LLM"
#
# llm = Yuan2_LLM('/root/autodl-fs/YuanLLM/Yuan2-2B-Mars-hf')
# print(llm("你是谁"))
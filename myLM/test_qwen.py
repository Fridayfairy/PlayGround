from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = '/mnt/34T/datasets/LLModels/Qwen/Qwen2.5-0.5B-Instruct'
config = AutoConfig.from_pretrained(model_path)
print(config)

# model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
print(model)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


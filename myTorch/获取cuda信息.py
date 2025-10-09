import torch

print(f"""
cuda 可用: {torch.cuda.is_available()}
cuda 设备数量: {torch.cuda.device_count()}
当前 cuda 设备: {torch.cuda.current_device()}
当前 cuda 设备名称: {torch.cuda.get_device_name(4)}
""")
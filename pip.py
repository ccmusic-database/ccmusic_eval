import subprocess

with open("requirements.txt") as f:
    for line in f:
        # 使用 subprocess 调用 pip 的 install 命令
        subprocess.check_call(["pip", "install", line.strip()])

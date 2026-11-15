import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kaggle_client import KaggleKernelClient

PROXY_URL = "https://kkb-production.jupyter-proxy.kaggle.net/k/369968446/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..Al1TtLL0RncPiKemxcw-3Q.L659Wq4JcJaMfzF3vNceDvXyvBjpRJ10rUHHOTxpJNrY9Hl7vQ4fqBvY8e4bH7ubpp0Sv4vlSiMdnIKzAG2YMJHkmSH9nZLv4XGCDXcV7J5z-bBXk8lscodIoqL4nU2Rvxq_-XGCihNKMut6WNNXWmxrIO5euEvJolgNF7Zn2Kp8PlO6aoctPqOYvtgsuEVCmNPzl5J1B7g132O-LtzXh5nCYZ_nBe4cT5oqKk1N8skHJatJIvINGjmjF0qxpGXT.5AjhK3tJuvpnzQ1raybYGQ/proxy"

client = KaggleKernelClient(PROXY_URL, kernel_id="914e109a-ee7e-4ce7-8eaa-b224eb70ee9a")
print("Connected to Kaggle Kernel ID (GPU 0):", client.kernel_id)

code = """
import sys, subprocess, os
os.chdir('/kaggle/working')

cmd = [
    sys.executable,
    'repo/scripts/train_crop_unet_1024_cldice.py',
    '--epochs', '15',
    '--batch_size', '16',
    '--crop_size', '512',
    '--encoder', 'resnet34',
    '--lr', '8e-4',
    '--alpha', '0.20',
    '--device_id', '0'
]

print("Launching Companion ResNet-34 clDice Training on Tesla T4 (GPU 0)...", flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(proc.stdout.readline, ''):
    print(line, end='', flush=True)
proc.stdout.close()
proc.wait()
print(f"\\nProcess finished with returncode: {proc.returncode}", flush=True)
"""

print("Starting execution on GPU 0...")
res = client.execute(code, timeout=7200)
print("\n--- Training Finished ---")
print("Status:", res.get("status"))
if res.get("error"):
    print("Error:", res.get("error"))

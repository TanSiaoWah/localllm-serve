# Local vLLM Setup on Windows with WSL2

This guide records the working setup for hosting a local model with **vLLM inside WSL2 Ubuntu** and calling it from **Python in VS Code on Windows** using the OpenAI Python SDK.

## Architecture

```text
Windows
│
├── VS Code
│   └── Python client
│       └── OpenAI Python SDK
│
│       HTTP request
│       http://localhost:8000/v1
│
└── WSL2
    └── Ubuntu 24.04
        └── vLLM
            └── Qwen/Qwen3-0.6B
                └── NVIDIA RTX 4070
```

---

# 1. Check WSL2

From Windows PowerShell or CMD:

```powershell
wsl -l -v
```

Example:

```text
NAME              STATE           VERSION
* Ubuntu          Running         2
  docker-desktop  Stopped         2
```

Make sure Ubuntu is using **WSL 2**.

Enter Ubuntu:

```powershell
wsl
```

Check the Ubuntu version:

```bash
cat /etc/os-release
```

Example used in this setup:

```text
Ubuntu 24.04.3 LTS
```

---

# 2. Check NVIDIA GPU Access Inside WSL

Inside Ubuntu:

```bash
nvidia-smi
```

The GPU should be visible.

Example setup:

```text
GPU: NVIDIA GeForce RTX 4070
VRAM: ~12 GB
```

You do not need to install a separate NVIDIA display driver inside WSL if `nvidia-smi` already works.

---

# 3. Create the vLLM Project Folder

Inside WSL:

```bash
cd ~
mkdir -p ~/vllm-test
cd ~/vllm-test
```

---

# 4. Install Required Ubuntu Packages

Update packages:

```bash
sudo apt update
```

Install curl and compiler tools:

```bash
sudo apt install -y curl build-essential
```

Verify GCC:

```bash
gcc --version
g++ --version
```

---

# 5. Install `uv`

Inside WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source ~/.bashrc
```

Check:

```bash
uv --version
```

---

# 6. Create the Python Environment for vLLM

Inside:

```text
~/vllm-test
```

create the environment:

```bash
uv venv --python 3.12 --seed --managed-python
```

Activate it:

```bash
source .venv/bin/activate
```

The terminal should now look similar to:

```text
(vllm-test) sw@DESKTOP:~/vllm-test$
```

---

# 7. Install vLLM

Inside the activated WSL environment:

```bash
uv pip install vllm --torch-backend=auto
```

Check the installed version:

```bash
vllm --version
```

Example from this setup:

```text
0.27.1
```

Check that PyTorch can access the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected result should contain:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

---

# 8. WSL-Specific vLLM Environment Variables

This setup required the following environment variables:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=gcc
export CXX=g++
```

Why:

- `VLLM_WSL2_ENABLE_PIN_MEMORY=1`
  - Enables the pinned-memory/UVA path needed by the vLLM V2 model runner under WSL2.

- `VLLM_USE_FLASHINFER_SAMPLER=0`
  - Prevents FlashInfer sampling from trying to JIT-compile CUDA code with `nvcc`.
  - This was needed because the WSL setup did not have a full CUDA toolkit at `/usr/local/cuda`.

- `CC=gcc`
  - Tells Triton which C compiler to use.

- `CXX=g++`
  - Tells Triton which C++ compiler to use.

## Make These Settings Permanent

Add them to `~/.bashrc`:

```bash
cat >> ~/.bashrc <<'EOF'

# vLLM on WSL2
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export CC=gcc
export CXX=g++
EOF
```

Reload:

```bash
source ~/.bashrc
```

---

# 9. Start the vLLM Server

Activate the WSL vLLM environment:

```bash
cd ~/vllm-test
source .venv/bin/activate
```

Start the model:

```bash
vllm serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

Keep this terminal open while using the model.

The server is now available at:

```text
http://localhost:8000
```

The OpenAI-compatible API base URL is:

```text
http://localhost:8000/v1
```

---

# 10. Check the Server from Windows

From Windows PowerShell or CMD:

```powershell
curl.exe http://localhost:8000/v1/models
```

A working response should include:

```json
{
  "id": "Qwen/Qwen3-0.6B"
}
```

At this point the model is successfully hosted by vLLM.

---

# 11. Create the Windows VS Code Python Client

Example project folder:

```text
D:\Github_Website\vllm-client
```

From CMD:

```cmd
cd /d D:\Github_Website\vllm-client
```

Create a Windows Python virtual environment:

```cmd
python -m venv .venv
```

## Important

There are now **two separate virtual environments**:

```text
WSL:
~/vllm-test/.venv
    -> contains vLLM and PyTorch
    -> runs the model server

Windows:
D:\Github_Website\vllm-client\.venv
    -> contains the Python client packages
    -> calls the vLLM server
```

Do not confuse these two environments.

---

# 12. Activate `.venv` in VS Code Using CMD

PowerShell may block `Activate.ps1` because of Windows execution policy.

The easiest option is to use **Command Prompt (CMD)** inside VS Code.

In VS Code:

```text
Terminal
→ Select Default Profile
→ Command Prompt
```

Then open a new terminal.

You should see something similar to:

```text
D:\Github_Website\vllm-client>
```

Activate the environment:

```cmd
.venv\Scripts\activate.bat
```

The prompt should become:

```text
(.venv) D:\Github_Website\vllm-client>
```

If VS Code is opened somewhere else first:

```cmd
cd /d D:\Github_Website\vllm-client
.venv\Scripts\activate.bat
```

---

# 13. Select the Virtual Environment in VS Code

Press:

```text
Ctrl + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Choose:

```text
D:\Github_Website\vllm-client\.venv\Scripts\python.exe
```

This tells VS Code to use the project's virtual environment.

---

# 14. Install the OpenAI Python SDK

With the Windows `.venv` activated:

```cmd
python -m pip install openai
```

You can verify:

```cmd
pip show openai
```

---

# 15. Create `test_vllm.py`

Create:

```text
test_vllm.py
```

with:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {
            "role": "user",
            "content": "Hello! Explain what vLLM is in simple terms."
        }
    ],
    max_tokens=200,
)

print(response.choices[0].message.content)
```

Run it:

```cmd
python test_vllm.py
```

If everything is working, the answer is generated locally by:

```text
VS Code Python
    ↓
OpenAI Python SDK
    ↓
localhost:8000
    ↓
vLLM inside WSL2
    ↓
Qwen3-0.6B
    ↓
RTX 4070
```

---

# 16. Interactive Python Example

You can also create an interactive client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    }
]

while True:
    question = input("You: ")

    if question.lower() in {"exit", "quit"}:
        break

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=messages,
        max_tokens=300,
    )

    answer = response.choices[0].message.content

    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    print(f"\nAI: {answer}\n")
```

The application is responsible for keeping the `messages` history and sending it back to the model on each request.

---

# 17. Normal Workflow After Everything Is Installed

## Terminal 1 — WSL: Start vLLM

```bash
wsl
```

Then:

```bash
cd ~/vllm-test
source .venv/bin/activate
```

Start the server:

```bash
vllm serve Qwen/Qwen3-8B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

Leave this terminal running.

## Terminal 2 — VS Code CMD: Run the Client

```cmd
cd /d D:\Github_Website\vllm-client
.venv\Scripts\activate.bat
python test_vllm.py
```

---

# 18. Useful Commands

## Check vLLM server

```powershell
curl.exe http://localhost:8000/v1/models
```

## Check GPU in WSL

```bash
nvidia-smi
```

Watch continuously:

```bash
watch -n 1 nvidia-smi
```

## Check which Python VS Code is using

CMD:

```cmd
where python
```

When `.venv` is activated, the first result should be:

```text
D:\Github_Website\vllm-client\.venv\Scripts\python.exe
```

## Check installed OpenAI SDK

```cmd
python -c "import openai; print(openai.__version__)"
```

## Exit the Windows virtual environment

```cmd
deactivate
```

## Exit the WSL virtual environment

```bash
deactivate
```

---

# 19. Troubleshooting Notes from This Setup

## `RuntimeError: UVA is not available`

Fix:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

## `Failed to find C compiler`

Install:

```bash
sudo apt install -y build-essential
```

Then:

```bash
export CC=gcc
export CXX=g++
```

## `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

For this local setup, disable the FlashInfer sampler:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This avoids requiring a full CUDA toolkit just for FlashInfer JIT sampling.

## PowerShell says `running scripts is disabled`

Instead of:

```powershell
.\.venv\Scripts\Activate.ps1
```

use CMD:

```cmd
.venv\Scripts\activate.bat
```

---

# Current Working Setup

```text
Host OS:
Windows

Linux:
WSL2 + Ubuntu 24.04.3 LTS

GPU:
NVIDIA GeForce RTX 4070 12 GB

vLLM:
0.27.1

Model:
Qwen/Qwen3-0.6B

vLLM endpoint:
http://localhost:8000

OpenAI-compatible endpoint:
http://localhost:8000/v1

Windows client:
VS Code + Python + OpenAI Python SDK
```

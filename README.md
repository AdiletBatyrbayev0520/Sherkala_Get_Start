# Sherkala – a Kazakh Language Model

This repository contains examples of using the Llama-3.1-Sherkala-8B-Chat model – a Kazakh language model.

## Environment Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   # Windows
   .venv/Scripts/activate

   # Linux/Mac
   source .venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   python -m pip install --upgrade pip
   ```

4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Uninstall the default CPU-only Torch to enable GPU usage:
   ```bash
   pip uninstall torch --yes
   ```

6. Install CUDA Toolkit version 12.8.1 for compatibility with your CUDA and PyTorch versions:
   ```
   https://developer.nvidia.com/cuda-12-8-1-download-archive
   ```

7. To install PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

8. Install `huggingface_hub` to access the inference API:
   ```bash
   pip install huggingface_hub
   ```

9. Log in to Hugging Face and request access to:
   ```
   https://huggingface.co/inceptionai/Llama-3.1-Sherkala-8B-Chat
   ```

10. Create an access token and save it somewhere:
    ```
    https://huggingface.co/settings/tokens
    ```

11. Authenticate with Hugging Face using your token:
    ```bash
    huggingface-cli login
    ```

## Basic Model Usage

Here’s a simple example showing how to load the model and generate responses:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "inceptionai/Llama-3.1-Sherkala-8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer.chat_template = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

' "
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>

' }}"
    "{% endif %}"
)

def get_response(text):
    conversation = [{"role": "user", "content": text}]

    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Generate a response
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=500,
        stop_strings=["<|eot_id|>"],
        tokenizer=tokenizer
    )

    # Decode and return the generated text (excluding the prompt)
    gen_text = tokenizer.decode(gen_tokens[0][len(input_ids[0]):-1])
    return gen_text

question = "Please give a brief overview of Kazakh history"
print(get_response(question))
```

## System Requirements

- Python 3.11 or higher  
- CUDA-compatible GPU (for GPU acceleration)  
- At least 16 GB of RAM (32 GB or more recommended)  

## Limitations

- The model may not always provide accurate or complete answers  
- Performance depends on available hardware  
- Additional configuration is required for multi-GPU setups  

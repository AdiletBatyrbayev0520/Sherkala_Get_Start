model_path = "inceptionai/Llama-3.1-Sherkala-8B-Chat"
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path="inceptionai/Llama-3.1-Sherkala-8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
device = "cuda" 

tokenizer.chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %} {% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


app = FastAPI()

@app.get("/")
def read_root():
    return {"This is Sherkala server": "You can generate text with Sherkala model"}

@app.post("/generate")
def generate(prompt: str):
    conversation = [
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt").to(device)

    # Generate a response
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=50,
        stop_strings=["<|eot_id|>"],
        tokenizer=tokenizer
        )

    # Decode and print the generated text along with generation prompt
    gen_text = tokenizer.decode(gen_tokens[0][len(input_ids[0]): -1])
    return {"text": gen_text}
 

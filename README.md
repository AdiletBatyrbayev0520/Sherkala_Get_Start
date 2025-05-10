# Sherkala - Казахстанская языковая модель

Этот репозиторий содержит примеры использования модели Llama-3.1-Sherkala-8B-Chat - языковой модели на казахском языке.

## Настройка окружения

1. Создайте виртуальное окружение Python:
```bash
python -m venv .venv
```

2. Активируйте виртуальное окружение:
```bash
# Windows
.venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate
```

3. Обновите pip:
```bash
python -m pip install --upgrade pip
```

4. Установите необходимые зависимости:
```bash
pip install -r requirements.txt
```

5. Удалите simple torch чтобы использовать GPU:
```bash
pip uninstall torch --yes
```

6. Установите CUDA ToolKit версии 12.8.1 для совместимости версии CUDA and Pytorch:
```bash
https://developer.nvidia.com/cuda-12-8-1-download-archive
```

7. Для использования CUDA by Pytorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

8. Скачай huggingface_hub для доступа к инференсу:
```bash
pip install huggingface_hub
```

9. Авторизуйся в huggingface и запроси доступ к:
```bash
https://huggingface.co/inceptionai/Llama-3.1-Sherkala-8B-Chat
```

9. Создай токен и сохрани его где то:
```bash
https://huggingface.co/settings/tokens
```


9. Авторизуйся в huggingface используя токен:
```bash
huggingface-cli login
```


## Базовое использование модели

Вот простой пример, демонстрирующий, как загрузить модель и генерировать ответы:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path="inceptionai/Llama-3.1-Sherkala-8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu" 

tokenizer.chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role']+'<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %} {% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


def get_response(text):
    conversation = [
        {"role": "user", "content": text}
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt").to(device)

    # Generate a response
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=500,
        stop_strings=["<|eot_id|>"],
        tokenizer=tokenizer
        )

    # Decode and print the generated text along with generation prompt
    gen_text = tokenizer.decode(gen_tokens[0][len(input_ids[0]): -1])
    return gen_text

question = 'Қазақстан тарихын қысқаша түсініктеме беріңіз'
print(get_response(question))
```

## Распределение модели между несколькими GPU

При наличии нескольких GPU, вы можете распределить модель между ними для эффективной работы с большими моделями. Для этого рекомендуется использовать метод `dispatch_model` из библиотеки `accelerate`:

```python
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "inceptionai/Llama-3.1-Sherkala-8B-Chat"

# 1. Загрузка пустой оболочки для определения размеров устройства
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

# 2. Автоматическое вычисление карты устройств для GPU
device_map = infer_auto_device_map(
    model,
    max_memory={0: "32GiB", 1: "32GiB", "cpu": "10GiB"},
    no_split_module_classes=["GPTJBlock"]  # опционально
)

# 3. Распределение параметров между устройствами
model = dispatch_model(
    model,
    device_map=device_map,
    offload_folder="offload"   # опционально: папка для выгрузки на CPU/диск
)

# 4. Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

> **Примечание**: Попытка использовать метод `model.parallelize(device_map)` вызовет ошибку `AttributeError: 'LlamaForCausalLM' object has no attribute 'parallelize'`, поэтому используйте `dispatch_model` как показано выше.

## Требования к системе

- Python 3.11 или выше
- CUDA-совместимый GPU (для ускорения на GPU)
- Минимум 16 ГБ ОЗУ (рекомендуется 32 ГБ или больше)

## Ограничения

- Модель может не всегда давать точные или полные ответы
- Производительность зависит от доступного оборудования
- Для использования с несколькими GPU требуется дополнительная настройка 
import readline

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_device

def main():
    device = get_device()
    model_id = "hinagiku/Qwen2.5-1.5B-Truthful-DPO"

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = []
    print("\n\n🍃 Assistant: Can I help you? \n")

    while True:
        prompt = input("👤 user: ").strip()

        if not prompt:
            continue
        
        messages.append({"role": "user", "content": prompt})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=1024)   
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages.append({"role": "assistant", "content": response})
        print(f"\n🍃 Assistant: {response}\n")

if __name__ == "__main__":
    main()
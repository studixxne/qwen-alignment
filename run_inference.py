import os

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from utils import *

@dataclass
class InferenceConfig:
    device: str = get_device()
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mode: str = 'sample'
    temperature: float = 0.9
    num_generations: int = 2
    do_sample: bool = True
    input_file: str = "./data/generated/synthetic_prompts.json"
    output_file: str = "./data/generated/qwen_responses"

def main():
    args = get_args(InferenceConfig)
    config = InferenceConfig(**vars(args))

    dataset = None

    if config.mode == 'eval':
        config.temperature = 0.0
        config.num_generations = 1
        config.do_sample = False
        dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]

    elif config.mode == 'sample':
        if not os.path.exists(config.input_file):
            raise FileNotFoundError(f"Not exist prompts file")
        
        with open(config.input_file, "r", encoding='utf-8') as f:
            dataset = json.load(f)
    
    else:
        raise ValueError("Mode Error!")
    
    prompts = [item["question"] for item in dataset]
    results = inference(config, prompts)

    final_outputs = []

    for item, res in zip(dataset, results):
        question = item['question']

        if config.mode == 'eval':
            final_outputs.append({
                "question": question,
                "model_answer": res[0],
                "correct_answers": item['correct_answers'],
                "incorrect_answers": item['incorrect_answers']
            })

        elif config.mode == 'sample':
            final_outputs.append({
                "question": question,
                "response_a": res[0],
                "response_b": res[1]
            })

    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    with open(config.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_outputs, f, ensure_ascii=False, indent=2)

def inference(config: InferenceConfig, prompts: list[str]) -> list[list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float16).to(config.device)
    logger = Logger(config)

    result = []
    total_prompt = len(prompts)

    logger.log_inference_header(config.mode, total_prompt, config.model_name)

    for step, prompt in tqdm(enumerate(prompts), desc=f"Qwen Processing [{config.mode.upper()}]"):
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors='pt').to(config.device)

        responses = []
        for _ in range(config.num_generations):
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, max_new_tokens=256, do_sample=config.do_sample, temperature=config.temperature)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            responses.append(response_text)

        result.append(responses)

        logger.log_inference_step(step+1, total_prompt, prompt, config.mode, responses)

    logger.log_inference_summary(total_prompt, config.output_file)
    
    return result

if __name__ == '__main__':
    main()
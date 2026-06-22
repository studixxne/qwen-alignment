import argparse
import json

from src import (
    run_prompt_generation, GenConfig,
    run_inference, InferenceConfig,
    run_dpo_llm_judge, JudgeConfig,
    dpo_train,  TrainConfig,
    evaluate_truthfulqa_mc, EvalConfig
)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Alignment Pipeline")
    parser.add_argument("--steps", nargs="+", default=["all"])
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--skip_base_eval", action="store_true")
    return parser.parse_known_args()[0]

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    args = parse_args()
    steps = args.steps
    run_all = "all" in steps
    cfg = load_config(args.config)

    print("🚀 Starting LLM Alignment Pipeline...")

    # =========================================
    # STEP 1: SYNTHETIC PROMPT GENERATION
    # =========================================
    if run_all or "gen_prompts" in steps:
        print('\n[STEP 1] Generating Synthetic Prompts...')
        run_prompt_generation(GenConfig(**cfg.get("gen_prompts", {})))

    # =========================================
    # STEP 2: BASE MODEL INFERENCE
    # =========================================
    if run_all or "run_inference" in steps:
        print('\n[STEP 2] Running Base Model Inference...')
        run_inference(InferenceConfig(**cfg.get("run_inference", {})))

    # =========================================
    # STEP 3: DPO PAIR GENERATION BY LLM JUDGING
    # =========================================
    if run_all or "gen_dpo_pairs" in steps:
        print('\n[STEP 3] Running LLM Judge...')
        run_dpo_llm_judge(JudgeConfig(**cfg.get("gen_dpo_pairs", {})))

    # =========================================
    # STEP 4: DPO TRAINING
    # =========================================
    if run_all or "dpo" in steps:
        print('\n[STEP 4] Starting DPO Training...')
        dpo_train(TrainConfig(**cfg.get("dpo", {})))

    # =========================================
    # STEP 5: MC2 EVALUATION
    # =========================================
    if run_all or "eval_mc" in steps:
        print('\n[STEP 5] Evaluating Model...')
        eval_cfg = cfg.get("eval_mc", {})

        if not args.skip_base_eval:
            print(" ➔ [1/2] Base Model Evaluation")
            base_cfg = eval_cfg.copy()
            base_cfg["peft_model_path"] = None
            evaluate_truthfulqa_mc(EvalConfig(**base_cfg))
        else:
            print(" ➔ [1/2] Base Model Evaluation Skipped")

        print(" ➔ [2/2] Aligned Model Evaluation")
        # 기본값을 ./checkpoints/best 에서 ./models/best_model 로 변경!
        aligned_cfg = eval_cfg.copy()
        aligned_cfg["peft_model_path"] = eval_cfg.get("peft_model_path", "./models/best_model")
        evaluate_truthfulqa_mc(EvalConfig(**aligned_cfg))

if __name__ == "__main__":
    main()
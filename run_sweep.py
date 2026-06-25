import os
import json
from itertools import product
import subprocess
import argparse
import wandb

def load_base_config():
    with open ("configs/config.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_temp_config(config_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[3e-6, 7e-6, 1e-5, 4e-5, 7e-5])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    return parser.parse_known_args()[0]

def main():
    args = parse_args()

    learning_rates = args.learning_rates
    betas = args.betas
    base_config = load_base_config()

    experiments = list(product(learning_rates, betas))
    total_exps = len(experiments)

    print(f"🔥 Total {total_exps} combinations")

    for i, (lr, beta) in enumerate(experiments):
        print("\n" + "="*65)
        print(f"🔥 [Experiment {i+1}/{total_exps}] LR: {lr} | Beta: {beta}")
        print("="*65)

        exp_config = base_config.copy()

        current_run_id = wandb.util.generate_id()
        model_dir_name = f"model_{lr}_{beta}"
        model_save_path = f"./models/{model_dir_name}"
        os.makedirs(model_save_path, exist_ok=True)

        exp_config["dpo"]["lr"] = lr
        exp_config["dpo"]["beta"] = beta
        exp_config["dpo"]["output_dir"] = model_save_path
        exp_config["dpo"]["run_name"] = model_dir_name
        exp_config["dpo"]["run_id"] = current_run_id
        exp_config["eval_mc"]["peft_model_path"] = model_save_path
        exp_config["eval_mc"]["run_name"] = model_dir_name
        exp_config["eval_mc"]["run_id"] = current_run_id

        sweep_config_path = f"{model_save_path}/run_config.json"
        save_temp_config(exp_config, sweep_config_path)

        try:
            command = [
                "python", "run_pipeline.py",
                "--steps", "dpo", "eval_mc",
                "--config", sweep_config_path
            ]
            subprocess.run(command, check=True)

        except subprocess.CalledProcessError as e:
            print(f"[Error] Experiment {i+1} failed.")
            continue

if __name__ == "__main__":
    main()
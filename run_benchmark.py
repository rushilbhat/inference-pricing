#!/usr/bin/env python3
import argparse
import subprocess
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("quantization", choices=config.QUANTIZATION_MAP.keys())
    parser.add_argument("tp", type=int, default=1)
    args = parser.parse_args()

    model_path = config.QUANTIZATION_MAP[args.quantization]
    
    print(f"\nModel={config.MODEL_FAMILY} | Mode={args.quantization} | TP={args.tp} | Cost=${config.SERVER_COST_PER_HR:.2f}/hr")

    try:
        print("\nStarting Server...")
        server_cmd = [
            "sudo", "docker", "run", "-d", "--name", "pricing-server",
            "--runtime", "nvidia", "--gpus", "all",
            "--network", "host",
            "-v", f"{subprocess.os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
            "--ipc=host",
            "vllm/vllm-openai:latest",
            "--model", model_path,
            "--port", str(config.PORT),
            "--tensor-parallel-size", str(args.tp)
        ]
        subprocess.run(server_cmd, check=True)

        client_env_vars = [
            f"-e", f"MODEL_FAMILY={config.MODEL_FAMILY}", 
            f"-e", f"MODEL_PATH={model_path}",
            f"-e", f"SEVER_TYPE={config.SERVER_TYPE}",
            f"-e", f"SERVER_COST_PER_HR={config.SERVER_COST_PER_HR}",
            f"-e", f"QUANTIZATION={args.quantization}",
            f"-e", f"TP={args.tp}",
            f"-e", f"PORT={config.PORT}"
        ]
        
        client_script = "pip install -q requests transformers numpy aiohttp datasets && python3 pricing_calculator.py"

        client_cmd = [
            "sudo", "docker", "run", "--rm", "--name", "pricing-client",
            "--network", "host",
            "-v", f"{subprocess.os.getcwd()}:/workspace",
            "-w", "/workspace"
        ] + client_env_vars + [
            "python:3.12-slim",
            "bash", "-c", client_script
        ]
        subprocess.run(client_cmd, check=True)
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nCleaning up...")
        subprocess.run(["sudo", "docker", "stop", "pricing-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["sudo", "docker", "rm", "pricing-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        print("Done.")

if __name__ == "__main__":
    main()
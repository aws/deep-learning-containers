import sys
import argparse
import logging
import os
import sys
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    logger.info("Training starts...")
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="huggan/flowers-102-categories")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()

    diffusers_version = "0.16.1"

    # download unconditional training script from diffusers
    branch = "v" + diffusers_version
    task = "unconditional_image_generation"
    script = "train_unconditional.py"

    url = (
        f"https://raw.githubusercontent.com/huggingface/diffusers/{branch}/examples/{task}/{script}"
    )
    os.system(f"wget {url}")

    # create default accelerate config
    os.system("accelerate config default")

    script_path = git_config["script"]

    # run accelerate command
    accelerate_cmd = f"accelerate launch {script_path} --dataset_name={args.dataset_name}" \
                    f" --resolution={args.resolution} --output_dir={args.output_dir}" \
                    f" --train_batch_size={args.train_batch_size} --num_epochs={args.num_epochs}" \
                    f" --gradient_accumulation_steps={args.gradient_accumulation_steps}"

    logger.info(f"Calling {accelerate_cmd}")
    os.system(accelerate_cmd)


if __name__ == "__main__":
    main()

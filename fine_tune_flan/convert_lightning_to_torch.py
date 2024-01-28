from train import FLANRS
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import argparse
import os

def convert_model(model_type, model_path, output_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_type, cache_dir=".cache")
    tokenizer.save_pretrained(output_dir)

    config = T5Config.from_pretrained(model_type, cache_dir=".cache", use_cache=True)
    t5 = T5ForConditionalGeneration.from_pretrained(model_type, cache_dir=".cache", config=config)
    pl_model = FLANRS.load_from_checkpoint(checkpoint_path=model_path, model=t5)
    pl_model.model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    # Check if model_dir is a directory
    if not os.path.isdir(args.model_dir):
        raise ValueError(f"The specified model_dir {args.model_dir} is not a directory or does not exist.")

    # Iterate through each file in the model_dir
    for file in os.listdir(args.model_dir):
        file_path = os.path.join(args.model_dir, file)
        if os.path.isfile(file_path) and file.endswith(".ckpt"):  # Check for checkpoint files
            output_dir = os.path.join(args.model_dir, ''.join(file.split('.')[:-1]))
            if os.path.isdir(output_dir):
                print(f"Model {file} has already been converted.")
                continue
            os.makedirs(output_dir, exist_ok=True)
            convert_model(args.model_type, file_path, output_dir)
            print(f"Converted model saved in {output_dir}")

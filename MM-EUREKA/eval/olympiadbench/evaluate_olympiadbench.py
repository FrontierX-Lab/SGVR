import argparse
import json
import os
import random
import time

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ds_collections = {
    "OE_MM_maths_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_MM_maths_en_COMP"},
    "OE_MM_physics_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_MM_physics_en_COMP"},
    "OE_TO_maths_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_TO_maths_en_COMP"},
    "OE_TO_physics_en_COMP": {"root": "Hothan/OlympiadBench", "split": "OE_TO_physics_en_COMP"},
}

SYSTEM_PROMPT = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(
            ds_collections[ds_name]["root"],
            ds_collections[ds_name]["split"],
            cache_dir=os.path.join(os.getcwd(), "data/OlympiadBench/"),
        )["train"]
        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            images_content = []
            if "image_1" in data_item:
                for i in range(1, 6):
                    if data_item[f"image_{i}"]:
                        images_content.append({"type": "image", "image": data_item[f"image_{i}"]})

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        *images_content,
                        {"type": "text", "text": data_item["question"]},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            if image_data:
                inputs.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image_data},
                    }
                )
            else:
                inputs.append(
                    {
                        "prompt": prompt,
                    }
                )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            if "image_1" in data_item:
                del data_item["image_1"]
                del data_item["image_2"]
                del data_item["image_3"]
                del data_item["image_4"]
                del data_item["image_5"]

            data_item["response"] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            id = data_item["id"]
            temp[id] = data_item

        print(f"Evaluating {ds_name} ...")
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{ds_name}_{time_prefix}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))

        cmd = f"python olympiadbench/extract_calculate.py --output_file {results_file}"
        print(cmd)
        # os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--datasets",
        type=str,
        default="OE_MM_maths_en_COMP,OE_MM_physics_en_COMP,OE_TO_maths_en_COMP,OE_TO_physics_en_COMP",
    )
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    llm = LLM(model=args.checkpoint, trust_remote_code=True, tensor_parallel_size=8, limit_mm_per_prompt={"image": 8})
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model()

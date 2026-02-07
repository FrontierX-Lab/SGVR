import argparse
import json
import os
import random
import time

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ds_collections = {"MathVerse_testmini": {"root": "AI4Math/MathVerse", "split": "testmini"}}


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        data = load_dataset(
            ds_collections[ds_name]["root"],
            ds_collections[ds_name]["split"],
            cache_dir=os.path.join(os.getcwd(), "data/MathVerse/"),
        )[ds_collections[ds_name]["split"]]

        inputs = []
        for idx, data_item in tqdm(enumerate(data)):
            data_item["query"] = data_item["query_cot"]
            image = data_item["image"]
            if args.prompt_template == "original":
                messages = [
                    {
                        "role": "system",
                        "content": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
                    },
                    {"role": "user", "content": "<image>\n" + data_item["query"]},
                ]
            elif args.prompt_template == "reasoning_instruct":
                messages = [
                    {
                        "role": "system",
                        "content": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
                    },
                    {
                        "role": "user",
                        "content": "<image>\nYou should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.\n"
                        + data_item["query"],
                    },
                ]
            elif args.prompt_template == "reasoning_pretrain":
                messages = [
                    {
                        "role": "system",
                        "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
                    },
                    {
                        "role": "user",
                        "content": f"<image>\nAnswer the following question: {data_item['query']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
                    },
                ]
            else:
                raise ValueError(f"Invalid prompt template: {args.prompt_template}")
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)

        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            del data_item["image"]
            data_item["response"] = model_output.outputs[0].text
            outputs.append(data_item)

        temp = {}
        for data_item in outputs:
            sample_index = data_item["sample_index"]
            temp[sample_index] = data_item

        print(f"Evaluating {ds_name} ...")
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{ds_name}_{time_prefix}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))

        cmd = f"python mathverse/extract_calculate.py --output_file {results_file}"
        print(cmd)
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="MathVerse_testmini")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt_template",
        type=str,
        choices=["original", "reasoning_instruct", "reasoning_pretrain"],
        default="original",
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_tokens = ["<|im_end|>\n".strip()]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    evaluate_chat_model()

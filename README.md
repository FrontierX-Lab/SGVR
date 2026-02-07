# Milestones over Outcome: Unlocking Geometric Reasoning with Sub-Goal Verifiable Reward

This is the official repository for the paper "Milestones over Outcome: Unlocking Geometric Reasoning with Sub-Goal Verifiable Reward".

![Framework](assets/framework.png)

## Overview

Multimodal Large Language Models (MLLMs) struggle with complex geometric reasoning tasks, largely because outcome-based "black box" supervision fails to distinguish between lucky guesses and rigorous deduction. To address this, we introduce a paradigm shift towards subgoal-level evaluation and learning.

### Key Contributions

1. **GeoGoal Benchmark**: The first multimodal geometry benchmark where intermediate sub-goals are formally verified and automatically checkable, introducing Skeleton Rate (SR), Skeleton Completion (SC) and Consistency Ratio (CR) as rigorous metrics for reasoning fidelity.

2. **SGVR Framework**: A reinforcement learning framework that leverages verifiable numeric sub-goals as critical reasoning milestones to provide dense supervision.

3. **Empirical Efficacy**: Experiments demonstrate that our proposed SGVR framework improves both final answer accuracy and intermediate reasoning quality, with strong cross-domain transfer to general reasoning tasks.

## Code and Data

The GeoGoal-SGVR dataset is now available on Hugging Face:

- **Dataset**: [GeoGoal-SGVR on Hugging Face](https://huggingface.co/datasets/carpe002/GeoGoal-SGVR)

### Installation

Our code is based on [MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA). To set up the environment:

```bash
cd MM-EUREKA
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

### Data Preparation

Download the GeoGoal-SGVR dataset from [Hugging Face](https://huggingface.co/datasets/carpe002/GeoGoal-SGVR) and prepare your training data in JSONL format. Each entry should follow this structure:

```json
{
  "id": "example_id",
  "question": "<image>\nYour geometric problem question here...",
  "answer": "Your answer",
  "message": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant...\"}, {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"<image>\\nYour question here...\"}, {\"type\": \"image\", \"image\": \"/path/to/your/image.png\"}]}]",
  "images": ["/path/to/your/image.png"]
}
```


### API Configuration

The SGVR reward function requires API access for sub-goal verification. Set the following environment variables:

```bash
export INFER_BASE_URL="your_api_base_url"
export INFER_API_KEY="your_api_key"
```

### Model Preparation

Prepare your SFT (Supervised Fine-Tuning) model checkpoint. The model should be compatible with the MM-EUREKA framework (currently supports Qwen2.5-VL and InternVL models).

### Training

Navigate to the training scripts directory and configure the following variables in the training scripts:

- `SFT_MODEL`: Path to your SFT model checkpoint
- `TRAIN_DATA`: Path to your training data JSONL file
- `OUTPUT_DIR`: Directory to save training outputs

We provide two training scripts:

1. **PPO Training**: `examples/scripts/run_sgvr_reward_ppo.sh`
2. **GRPO Training**: `examples/scripts/run_sgvr_reward_grpo.sh`

Example usage:

```bash
cd MM-EUREKA/examples/scripts
# Edit the script to set SFT_MODEL, TRAIN_DATA, and OUTPUT_DIR
bash run_sgvr_reward_ppo.sh
# or
bash run_sgvr_reward_grpo.sh
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{chen2026milestonesoutcomeunlockinggeometric,
      title={Milestones over Outcome: Unlocking Geometric Reasoning with Sub-Goal Verifiable Reward}, 
      author={Jianlong Chen and Daocheng Fu and Shengze Xu and Jiawei Chen and Yuan Feng and Yue Yang and Junchi Yan and Hongyuan Zha and Renqiu Xia},
      year={2026},
      eprint={2601.05073},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.05073}, 
}
```

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.



<div align="center">
    <img alt="MM-Eureka logo" src="./docs/logo.png" style="height: 200px;" />
</div>


<div align="center">

# MM-EUREKA

</div>

<div align="center">
<p align="center">
  📖<a href="https://jagged-court-d9d.notion.site/MM-Eureka-Qwen-1c13cc5a384880ffbd2de24e1dee052d">Report</a> |
  📊<a href="https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset">Datasets</a> |
  🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B">MM-Eureka-Qwen-7B</a> |
</p>
</div>

<hr>
<div align="center">
<p style="text-align: center;">MM-EUREKA: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning<p>
</div>
<hr>
<div align="center">
<a href="https://jagged-court-d9d.notion.site/MM-Eureka-Qwen-1c13cc5a384880ffbd2de24e1dee052d">[[Report Link]]</a>
</div>



## 🎯Overview

We present **MM-Eureka-Qwen**, a multimodal reasoning model that successfully extends large-scale rule-based reinforcement learning (RL) to multimodal reasoning. Compared to the previous version of MM-EUREKA based on InternVL, we have made improvements in model architecture, algorithms, and data. Using only out-of-domain training data, MM-Eureka-Qwen achieves significant improvements over Qwen2.5-VL-Instruct-7B across multiple benchmarks (e.g. MathVista 73.0). 

**Core Improvements:**

1. We further iterate the codebase to support algorithms including Online Filter, [ADORA](https://github.com/ShadeCloak/ADORA?tab=readme-ov-file), and [DAPO](https://arxiv.org/abs/2503.14476).
2. We expand our K12 dataset from MM-EUREKA-Dataaset, from 8k to 15k high-quality K12 samples.
3. We train the MM-Eureka-Qwen-7B model, achieving better results with significantly lower cost than the previous version.

🔥We open-source our complete pipeline to foster further research in this area. We release all our codes, models, data, etc. at [MM-EUREKA-Qwen](https://github.com/ModalMinds/MM-EUREKA/tree/qwen).

## 🗞️ News

- **[2025/03/27]** We released `MM-Eureka-Qwen`.
  - 📖 Report: [MM-Eureka-Qwen-Report](https://jagged-court-d9d.notion.site/MM-Eureka-Qwen-1c13cc5a384880ffbd2de24e1dee052d) 
  - 🤗 Model: [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)
  - 📊 Dataset: [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)
  - 🚀Code: [MM-Eureka-Qwen-Code](https://github.com/ModalMinds/MM-EUREKA/tree/qwen)

- **[2025/03/07]** We released `MM-Eureka`.
  - 📖 Paper: [MM-Eureka-paper](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf)
  - 🤗 Model: [MM-Eureka-8B](https://huggingface.co/FanqingM/MM-Eureka-8B) & [MM-Eureka-Zero-38B](https://huggingface.co/FanqingM/MM-Eureka-Zero-38B)
  - 📊 Dataset: [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)
  - 🚀Code: [MM-Eureka-Code](https://github.com/ModalMinds/MM-EUREKA/tree/internvl)

## 🚀 Features

This repository is built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), introducing several key enhancements:

- **Multimodal RFT Support**: Extends OpenRLHF to incorporate **vision-language models (VLMs)**, currently supporting **InternVL**, enabling multimodal reasoning capabilities.
  - Currently support **RLOO**, **REINFORCE++**, **GRPO** training using Ray.
  - vLLM integration and distributed training.
  - Support hybrid engine (`--colocate_all_models`, `--vllm_enable_sleep`).
- **Better Rule-based Reward support**: Better training visualization for Rule-based Rewards (i.g. Format Reward, Accuracy Reward, Repetition Penalty)
- **Enhanced Online Filtering**: Filtering out experiences based on Accuracy Reward during training as in [PRIME](https://github.com/PRIME-RL/PRIME)
  - Use `--enable_accuracy_filter`, `--freezing_filter_steps`, `--accuracy_lower_bound`, `--accuracy_upper_bound` to control the behavior of online accuracy filter.
- **ADORA**: Enable Adaptive Online Rollout Adjustment by using `--use_adora` and `--adora_lamda` as in [ADORA](https://five-stetson-b51.notion.site/Training-Reasoning-Model-with-Dynamic-Advantage-Estimation-on-Reinforcement-Learning-1a830cc0904681fa9df3e076b6557a3e).
- **DAPO**: You can use `--use_dapo` to enable DAPO loss during training as in [DAPO](https://arxiv.org/abs/2503.14476).


## 🤖 Models

Based on the key factors identified by https://github.com/ModalMinds/MM-EUREKA for achieving stable training, we enhanced the model, dataset, and algorithmic modules. Specifically, we maintained the strategy of omitting the KL divergence term and applying data filtering, while implementing the following critical modifications:

- The base model was upgraded from InternVL2.5-8B-Instruct to the more powerful Qwen2.5-VL-7B-Instruct.
- The Vision Transformer (ViT) module was frozen during training.
- The underlying RL algorithm was replaced with [GRPO](https://arxiv.org/pdf/2402.03300), instead of the previously used RLOO.
- The data filtering strategy was transitioned from an offline approach to an online approach.
- Additional data from the K12 dataset was collected, expanding the total dataset size to 15,000 samples.

Finally, MM-EUREKA-Qwen achieves 73.0 on MathVista, surpassing the original Qwen2.5-VL-7B-Instruct by 4.8%.

|                             | Base Model                 | MathVista       | MathVerse       | MathVision | OlympidBench  | K12              |
| --------------------------- | -------------------------- | --------------- | --------------- | ------- | ------------- | ---------------- |
| Qwen2.5-VL-7B-Instruct      | -                          | 68.2            | 47.9            | 25.4    | 15.3          | 36.0             |
| InternVL2.5-VL-8B-Instruct  | -                          | 64.4            | 39.5            | 19.7    | 8.0           | 24.8             |
| InternVL2.5-VL-38B-Instruct | -                          | 71.9            | 49.4            | 31.8    | 29.3          | 37.2             |
| MM-EUREKA-InternVL-8B       | InternVL2.5-VL-7B-Instruct | 67.1            | 40.4            | 22.2    | 8.6           | 27.0             |
| MM-EUREKA-Qwen-7B           | Qwen2.5-VL-7B-Instruct     | **73.0 (+4.8)** | **50.3 (+2.4)** | 26.9 (+1.5) | 25.3（+10.0） | **48.6 (+12.6)** |

- 🤗 [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)

## 🏁 Getting Started

### 📦 Installation

```shell
git clone https://github.com/ModalMinds/MM-EUREKA.git
git checkout qwen
cd MM-EUREKA
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

### 📂 Data Preparation

You can download our training data from [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)

Once downloaded, refer to the section below for additional data formation.

#### Custom dataset

For custom dataset, format your data in to a JSONL file,  where each entry is a dictionary organized in the following format. 

```json
{
  "id": "0",
  "message": "[{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file:///path/to/your/image.jpg\"}, {\"type\": \"text\", \"text\": \"How many cats in the image?\"}]}]",
  "answer": "gt that could be parsed and verified by math_verify"
}
```

### 🌐 Start Training

Before starting your own training, ensure that the paths in the provided training scripts are correctly set and that environment variables like `$MASTER_ADDR` and `$NODE_RANK` are properly configured.

**start MM-Eureka-Qwen-7B training**

- for single node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_single_node.sh
  ```

- for multiple node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_multi_node.sh
  ```

## ⭐ Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=ModalMinds/MM-EUREKA&type=Date)](https://star-history.com/#ModalMinds/MM-EUREKA&Date)

## 🤝 Contribution

MM-Eureka is stil under active development, if you want to contribute, please feel free to make a pull request or create an issue.

Please refer to `CONTRIBUTING.md` before you dive in！

## 📬 Contact

If you have any questions or would like to engage with our community, feel free to scan the QR code below to join our WeChat group.

<div align="center">
<img alt="MM-Eureka logo" src="docs/wechat_qr.png" style="height: 400px;" />
</div>

## 🎓 Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1) and [vLLM](https://github.com/vllm-project/vllm). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [InternVL](https://github.com/OpenGVLab/InternVL) and [QwenVL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source techniques and base models, which have enabled us to further our exploration.

## 📜 Citation
```
@article{meng2025mm,
  title={MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning},
  author={Meng, Fanqing and Du, Lingxiao and Liu, Zongkai and Zhou, Zhixiang and Lu, Quanfeng and Fu, Daocheng and Shi, Botian and Wang, Wenhai and He, Junjun and Zhang, Kaipeng and others},
  journal={arXiv preprint arXiv:2503.07365},
  year={2025}
}
```

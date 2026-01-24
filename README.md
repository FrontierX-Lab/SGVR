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

**Code will be released within one month.**


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



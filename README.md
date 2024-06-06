# LLM-guards-RLHF

This is the source code for our paper:
[Safeguarding LLM-Applications: Specify or Train?]

In this paper, we aim to (1) evaluate whether self-training pipelines can have similar or outperform the accuracy of existing guardrails. (2) Compare the engineering effort involved in developing these guardrails with existing solutions like NeMo guardrails versus the proposed self-training pipelines.

## Usage

### 1. Datasets

Please download the training and testing data in folder `./train_test_data`. 

### 2. Models 
For each pipeline, we fine-tune three models:
1. The Supervised Fine-Tuning (SFT) model; Llama-3 -- run create_SFT.py 
2. DistilBERT as the reward model -- run bert_classifier_train.py
3. Optimise the SFT model with the PPO algorithm using the reward model -- run ppo_reward.py for moderation and off-topic pipelines, and ppo_reward_unified_guard.py for the unified pipeline. 

## Pipelines Results

Topical pipeline:
|   Method          |     % of off-topic prompts blocked | % of relevant prompts allowed| 
| ------------------ |---------------- | ---------------- |
| Llama-3            |    98.7% |     99.4%     |  
| mistral |      99.42%     | 99.28% |
| gemma |     25.71%     | 70.71% |
| NeMo Guardrails |  81%         | NA|

Moderation pipeline:
|   Method         |     % of harmful prompts blocked    | % of helpful prompts allowed  | 
| ------------------ |---------------- | ---------------- |
| Llama-3 |    97.86%       |  78.14% |
| mistral |    95.43%    | 93.29% |
| gemma |      97%    | 28.86% |
| NeMo Guardrails |  78%         | 97% |

Unified pipeline:
|   Method         |    % of relevant prompts allowed  | % of off-topic prompts blocked | 
| ------------------ |---------------- | ---------------- |
| Llama-3 |   98%      | 93.43%  | 94.71% |
| mistral |  97.57%  | 98.86% | 85.86% |
| gemma |     63.29%   | 69.43% | 90.14% |


## Citation

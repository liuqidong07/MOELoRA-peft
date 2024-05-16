# When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications

This is the implementation of the SIGIR'24 paper "When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications".

__If any quetions, you can firstly refer to the issues in [initial repo](https://github.com/liuqidong07/MOELoRA-peft).__

## Running

You can implement our model according to the following steps:

1. The handle dataset should be put into `./data/`
2. Put all files of ChatGLM-6B into the folder `resources/chatglm-6b/` and replace the the `modeling_chatglm.py` by our `modeling_chatglm.py` in this folder.
3. Install the necessary packages. Run the command:
   ```
   pip install -r requirements.txt
   ```
4. To train the MOELoRA and generate the answers to test, please run the command:
   ```
   bash ./experiments/moelora.bash
   ```
5. Finally, you can run and configure the `results/evaluate.ipynb` to get the evaluation scores

## Requirements

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Tesla V100 32GB
  - Cuda: 10.2
  - Driver version: 440.95.01
  - CPU: Intel Xeon Gold 6133
- Software:
  - Python: 3.9.5
  - Pytorch: 1.12.0+cu102
  - transformers: 4.28.1
  - deepspeed: 0.9.4

You can also try the `environment.yml` to install the environment.

## Citation

If the code and the paper are useful for you, it is appreciable to cite our paper:

```
@article{liu2023moelora,
  title={Moelora: An moe-based parameter efficient fine-tuning method for multi-task medical applications},
  author={Liu, Qidong and Wu, Xian and Zhao, Xiangyu and Zhu, Yuanshao and Xu, Derong and Tian, Feng and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2310.18339},
  year={2023}
}
```

## Thanks

The code refers to the repo **[PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)**.

Besides, thank **[lhyscau](https://github.com/lhyscau)** to help me refine the code.

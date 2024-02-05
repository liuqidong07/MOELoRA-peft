# MOELoRA: An MOE-based Parameter Efficient Fine-Tuning Method for Multi-task Medical Applications

This is the implementation of the paper "MOELoRA: An MOE-based Parameter Efficient Fine-Tuning Method for Multi-task Medical Applications".

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

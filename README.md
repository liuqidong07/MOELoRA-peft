# When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications

This is the implementation of the paper "When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications".

You can implement our model according to the following steps:

1. The handle dataset should be put into `./data/<dataset>/handled/`
2. Install the necessary packages. Run the command:
   ```
   pip install -r requirements.txt
   ```
3. To train the MOELoRA, please run the command:
   ```
   bash ./experiments/moelora.bash
   ```
4. Finally, you can run the following bash to test the the MOELoRA:
   ```
   bash ./experiments/test.bash
   ```

# -*- encoding: utf-8 -*-
'''
@File    :   chatglm2.py
@Time    :   2023/08/09 17:47:24
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import json



class chatglm2_train(object):

    def __init__(self, data_args, model_args, prompt_column, 
                 response_column, history_column, prefix, tokenizer, task=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task

    def __call__(self, examples):

        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("datasets/pre_data/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                history = examples[self.history_column][i] if self.history_column is not None else None
                prompt = self.tokenizer.build_prompt(query, history)

                prompt = self.prefix + prompt
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                            max_length=self.data_args.max_source_length)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                            max_length=self.data_args.max_target_length)
                
                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                if len(b_ids) > self.data_args.max_target_length - 2:
                    b_ids = b_ids[: self.data_args.max_target_length - 2]

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)
                #input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                #labels = labels + [self.tokenizer.pad_token_id] * pad_len
                if self.data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

                if self.task:
                    task_id = task_dict[examples['task_dataset'][i]]
                    model_inputs["task_id"].append(task_id)

        return model_inputs



class chatglm2_eval(object):

    def __init__(self, data_args, model_args, prompt_column, 
                 response_column, history_column, prefix, tokenizer, task=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task

    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        if self.task:
            task_id = []
            task_dict = json.load(open("datasets/pre_data/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if self.examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                history = examples[self.history_column][i] if self.history_column is not None else None
                prompt = self.tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[self.response_column][i])
            
            if self.task:
                task_id.append(task_dict[examples['task_dataset'][i]])

        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs,
                                    max_length=self.data_args.max_source_length,
                                    truncation=True,
                                    padding=True)
        labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        if self.task:
            model_inputs["task_id"] = task_id

        return model_inputs







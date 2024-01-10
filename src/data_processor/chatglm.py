# -*- encoding: utf-8 -*-
# here put the import lib
import json



class chatglm1_train(object):
    
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        if self.task:
            model_inputs["task_id"] = []
            task_dict = json.load(open("datasets/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]


        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i] and examples[self.response_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]

                if self.history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = self.prefix + prompt
                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                if len(b_ids) > self.data_args.max_target_length - 2:
                    b_ids = b_ids[: self.data_args.max_target_length - 2]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                if self.model_args.model_name_or_path != "resources/chatglmv2":
                    context_length = input_ids.index(self.tokenizer.bos_token_id)
                else:
                    context_length = len(a_ids) ### For ChatGLMv2
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position+1:]
                
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



class chatglm1_eval(object):
    
    def __init__(self, data_args, model_args, prompt_column, 
                response_column, history_column, prefix, tokenizer, 
                task=False, department=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.department = department

        

    def __call__(self, examples):
    
        max_target_length = self.data_args.max_target_length
        inputs, targets = [], []

        if self.task:
            task_id = []
            task_dict = json.load(open("datasets/task_dataset.json", "r"))
            task_dict = task_dict["str2id"]

        for i in range(len(examples[self.prompt_column])):
            if not examples[self.response_column][i]:
                targets.append("filled in !")
            else:
                targets.append(examples[self.response_column][i])

            if examples[self.prompt_column][i]:
                query = examples[self.prompt_column][i]
                if self.history_column is None or len(examples[self.history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[self.history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                inputs.append(prompt)

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
        


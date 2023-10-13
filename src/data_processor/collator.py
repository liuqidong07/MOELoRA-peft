# -*- encoding: utf-8 -*-
# here put the import lib
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import transformers

IGNORE_INDEX = -100


@dataclass
class LongestSequenceCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    task_flag: bool
    depart_flag: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        if self.task_flag:
            task_id = [instance["task_id"] for instance in instances]
            task_id = torch.LongTensor(task_id)

            if self.depart_flag:    # if add the department and entity
                depart = [instance["depart"] for instance in instances]
                depart = torch.LongTensor(depart)

                entity = [instance["entity"] for instance in instances]
                entity = torch.stack(entity)
                entity = torch.LongTensor(entity)

                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    task_id=task_id,
                    depart=depart,
                    entity=entity,
                )

            return dict(
                input_ids=input_ids,
                labels=labels,
                task_id=task_id,
            )

        return dict(
            input_ids=input_ids,
            labels=labels,
        )


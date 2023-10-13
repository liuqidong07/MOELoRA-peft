# -*- encoding: utf-8 -*-
# here put the import lib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from src.MLoRA.main import main
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from src.MLoRA.arguments import ModelArguments, DataTrainingArguments


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    main(parser)


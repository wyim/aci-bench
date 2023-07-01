
import argparse
import pandas as pd
import math
import os
import random
from pathlib import Path
import json
import nltk
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from datasets import load_metric, load_dataset, Dataset
from tqdm import tqdm
import nlp
from nlp import load_dataset
import logging
from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainingArguments,AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,LongformerTokenizer, EncoderDecoderModel, Trainer, LEDTokenizer, LEDForConditionalGeneration
import numpy as np
import evaluate
from datetime import datetime

# set the gpu device
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

#from transformers.data.data_collator import tf_default_data_collator

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--global_attention",
        type=int,
        default=128,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument("--testing_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--testing_dir_contrast", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--val_min_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def get_dataset(filename,tokenizer,args):
    all_df = []
    with open(filename,encoding="utf-8") as f:
         dataset=json.load(f)["data"]
    output={"src":[],"tgt":[]}
    for task in dataset:
        output["src"].append(task["src"])
        output["tgt"].append(task["tgt"])
    all_df = pd.DataFrame.from_dict(output)
    dataset = Dataset.from_pandas(all_df)
    return dataset
    
def main():
    logger = logging.getLogger(__name__)
    args = parse_args()
    if not os.path.isdir(args.output_dir): #
            os.mkdir(args.output_dir)
    logging.basicConfig(filename=args.output_dir+'log.log',level=logging.INFO,filemode='w')
    
    #model #
    if args.model_name_or_path=="allenai/led-large-16384":
        tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384")
        model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-large-16384", gradient_checkpointing=True, use_cache=False)
    elif (args.model_name_or_path in ["patrickvonplaten/led-large-16384-pubmed","hyesunyun/update-summarization-bart-large-longformer"]) \
        or ("fu" in args.model_name_or_path):
        if "fu" in args.model_name_or_path:
            if "LED_1024" in args.model_name_or_path:
                tokenizer=LEDTokenizer.from_pretrained("allenai/led-large-16384")
            if "LED_pubmed_1024" in args.model_name_or_path:
                tokenizer=LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
        else:
            tokenizer=LEDTokenizer.from_pretrained(args.model_name_or_path)
        model=LEDForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError('model not identified')
    #LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    
    #LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name_or_path, "roberta-base")
    # tokenizer = LongformerTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        # model_inputs = tokenizer(examples[args.summary_column],#args.text_column], #cheat
        #                          max_length=args.max_length, 
        #                          truncation=True,
        #                          padding="max_length")

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            model_inputs=tokenizer([t for t in examples[args.text_column]], 
                               max_length=args.max_length, padding="max_length", truncation=True)
            labels = tokenizer([t for t in examples[args.summary_column]], 
                               max_length=args.max_target_length, padding="max_length", truncation=True)
        labels["input_ids"]=[[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in labels["input_ids"]]
        model_inputs["decoder_input_ids"] = labels["input_ids"] 
        model_inputs["labels"] = labels["input_ids"] 
        model_inputs["global_attention_mask"] = [[1 if i < args.global_attention else 0 for i in range(sequence_length)] for sequence_length in len(model_inputs["input_ids"]) * [args.max_length]]
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        #print("global attention: {}".format(args.global_attention))
        return model_inputs
    # load train and validation data
    #train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
    #val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")
    #train_dataset=get_dataset(args.train_file,tokenizer,args)
    train_dataset=get_dataset(args.train_file,tokenizer,args).map(preprocess_function, 
                                                                     batched=True,
                                                                     batch_size=args.per_device_eval_batch_size)
    #print(train_dataset)
    val_dataset=get_dataset(args.validation_file,tokenizer,args).map(preprocess_function, 
                                                                     batched=True,
                                                                     batch_size=args.per_device_eval_batch_size)
    #print(val_dataset)
    test_dataset=get_dataset(args.test_file,tokenizer,args).map(preprocess_function, 
                                                                     batched=True,
                                                                     batch_size=args.per_device_eval_batch_size)
    #print(test_dataset)
    # enable gradient checkpointing for longformer encoder
    #model.encoder.config.gradient_checkpointing = True

    # set decoding params
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id=tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.max_length = args.max_length
    model.config.min_length = args.val_min_target_length
    model.config.no_repeat_ngram_size = 1
    model.early_stopping = True
    model.length_penalty = 2.0
    model.num_beams = args.num_beams

    encoder_length = args.max_source_length
    decoder_length = args.max_target_length

    nltk.data.find("tokenizers/punkt")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n ".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n ".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        rouge = evaluate.load('rouge')
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # decoded_preds = [" ".join(tokenizer.convert_ids_to_tokens(pred)) for pred in preds]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = [" ".join(tokenizer.convert_ids_to_tokens(label)) for label in labels]
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result =rouge.compute(predictions=decoded_preds,references=decoded_labels)
        logging.info(result)
        output=[]
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        for p,t in zip(decoded_preds, decoded_labels):
            output.append({"true":t,"pred":p})
        with open(args.output_dir+"prediction-{}.json".format(current_time),"w") as f:
                json.dump(output, f,indent=4)
        return result
    
    training_args = Seq2SeqTrainingArguments( #
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=False if args.only_test else True,
        do_eval=True,
        do_predict=True,
        learning_rate=args.learning_rate,
        overwrite_output_dir=True,
        save_total_limit=3,
        num_train_epochs=args.num_train_epochs,
        generation_max_length=args.max_target_length,
        evaluation_strategy="epoch", 
        predict_with_generate=True,
#        eval_delay=args.num_train_epochs,
        eval_delay=20,
        fp16=True,
        seed=args.seed,
        weight_decay=0.01,
        warmup_ratio=0.1
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instantiate trainer
    trainer = Seq2SeqTrainer(#
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset#val_dataset,#train_dataset
    )

    # start training
    if not args.only_test:
        trainer.train()

    trainer.evaluate()

    #save model
    trainer.save_model(args.output_dir)
if __name__ == "__main__":
    main()
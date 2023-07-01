import argparse
import logging
import math
import os
import random
from pathlib import Path
import json
import shutil
from glob import glob
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric,concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    LEDTokenizer, 
    LEDForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
    EncoderDecoderConfig,
    LongformerTokenizer, 
    EncoderDecoderModel,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed
)
from bert_score import score
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version

# set the gpu device
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


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
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--train_file2", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file2", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file3", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file4", type=str, default=None, help="A csv or a json file containing the training data."
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
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
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


def main():
    args = parse_args()
    if not os.path.isdir(args.output_dir): #
        os.mkdir(args.output_dir)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        filename=args.output_dir+'log.log', #
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            assert os.path.isfile(args.train_file), args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            assert os.path.isfile(args.validation_file),  args.validation_file
        if args.train_file2 is not None:
            data_files["train2"] = args.train_file2
        if args.test_file is not None:
            data_files["test"] = args.test_file
            assert os.path.isfile(args.test_file), args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        if "best_model" in args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.tokenizer_name)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info(config)
    if "longformer" in args.model_name_or_path.lower():
        #args.model_name_or_path=="yikuan8/Clinical-Longformer":
        #tokenizer = LongformerTokenizer.from_pretrained(args.model_name_or_path) 
        tokenizer = LEDTokenizer.from_pretrained(args.model_name_or_path)
        #AutoTokenizer.from_pretrained(args.model_name_or_path)
        # #
    elif args.tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        #tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "longformer" in args.model_name_or_path.lower():
        #model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name_or_path, "roberta-base")
        model = LEDForConditionalGeneration.from_pretrained(args.model_name_or_path)
        #AutoModel.from_pretrained(args.model_name_or_path)
        # #
        #AutoModelForMaskedLM
    elif args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)
    
    
    # commented this
    #model.resize_token_embeddings(len(tokenizer))
    #if model.config.decoder_start_token_id is None:
    #    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    logging.info("max_length:{}; Pad to max_length".format(args.max_target_length,args.pad_to_max_length))
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, 
                                 max_length=args.max_source_length, 
                                 padding=padding, 
                                 truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]
    
    if args.train_file2: #combine two files
        train_dataset2 = concatenate_datasets([processed_datasets["train"],
                                              processed_datasets["train2"]])

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, num_workers=0,
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, num_workers=0, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, num_workers=0, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader)
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("metric/rouge.py")

    # Train!
    if not args.only_test:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        best_val_score = 0
        best_epoch = -1
        for epoch in range(args.num_train_epochs):
            model.train()

            #######only for ablation, load the previous best model.
            # if epoch >=15:
            #     train_dataloader=eval_dataloader
            #     model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir+f'/{best_epoch}', config=config)
            # ##
            for step, batch in enumerate(train_dataloader):
                # print(batch.keys)
                # print(batch['input_ids'].shape)
                # print(batch['labels'].shape)
                # print(batch['decoder_input_ids'].shape)
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break


            model.eval()
            if args.val_max_target_length is None:
                args.val_max_target_length = args.max_target_length

            gen_kwargs = {
                "max_length": args.val_max_target_length if args is not None else config.max_length,
                "min_length": args.val_min_target_length,
                "num_beams": args.num_beams,
            }
            if epoch >=0: #evaluation
                to_dump=[] #
                
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        if args.model_name_or_path=="yikuan8/Clinical-Longformer": #
                            outputs = model(batch["input_ids"], 
                                            attention_mask=batch["attention_mask"]
                                            **gen_kwargs)
                            start_logits = outputs.start_logits
                            end_logits = outputs.end_logits
                            all_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0].tolist())
                            answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
                        else:
                            generated_tokens = accelerator.unwrap_model(model).generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                **gen_kwargs,
                            )

                        generated_tokens = accelerator.pad_across_processes(
                            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                        )
                        labels = batch["labels"]
                        if not args.pad_to_max_length:
                            # If we did not pad to max length, we need to pad the labels too
                            labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                        labels = accelerator.gather(labels).cpu().numpy()

                        if args.ignore_pad_token_for_loss:
                            # Replace -100 in the labels as we can't decode them.
                            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                        if isinstance(generated_tokens, tuple):
                            generated_tokens = generated_tokens[0]
                        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                        #
                        
                        decoded_source = tokenizer.batch_decode(batch["input_ids"],skip_special_tokens=True)
                        
                        for source,true,pred in zip(decoded_source,decoded_labels,decoded_preds):
                            sample={"source":source,"true":true,"pred":pred}
                            to_dump.append(sample)
                
                with open(args.output_dir+"prediction_epoch{}.json".format(epoch),"w") as f:
                    json.dump(to_dump, f,indent=4)
                
                result = metric.compute(use_stemmer=True)
                # Extract a few results from ROUGE
                result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

                result = {k: round(v, 4) for k, v in result.items()}

                if np.mean(list(result.values())) > best_val_score:
                    best_val_score = np.mean(list(result.values()))
                    best_epoch = epoch

                logger.info(f'At epoch {epoch}, the evaluation result is {result}, the best score is {best_val_score} at epoch {best_epoch}')
            #
                
            # if epoch - best_epoch >= 2:
            #     logger.info(f'Early stop at epoch {epoch}.')
            #     break
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            os.makedirs(args.output_dir+f'/{epoch}/', exist_ok=True)
            unwrapped_model.save_pretrained(args.output_dir+f'/{epoch}', save_function=accelerator.save)
            
            if args.push_to_hub:
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )
    accelerator.wait_for_everyone()
    if args.only_test:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.testing_dir, config=config)
        model_contrast = AutoModelForSeq2SeqLM.from_pretrained(args.testing_dir_contrast, config=config)
        model_contrast = accelerator.prepare(model_contrast)
        model_contrast.eval()
    elif best_epoch!=-1:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir+f'/{best_epoch}', config=config)
        os.rename(args.output_dir+f'/{best_epoch}',args.output_dir+f'/best_model')
    model = accelerator.prepare(model)
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "min_length": args.val_min_target_length,
        "num_beams": args.num_beams,
    }
    all_cands = []
    all_refs = []
    to_dump=[]
    for step, batch in tqdm(enumerate(test_dataloader)):
            #
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                # if args.only_test:
                #     generated_tokens_contrast = accelerator.unwrap_model(model_contrast).generate(
                #         batch["input_ids"],
                #         attention_mask=batch["attention_mask"],
                #         **gen_kwargs,
                #     )
                #     generated_tokens_contrast = accelerator.pad_across_processes(
                #         generated_tokens_contrast, dim=1, pad_index=tokenizer.pad_token_id
                #     )
                #     generated_tokens_contrast = accelerator.gather(generated_tokens_contrast).cpu().numpy()
                #     contexts = accelerator.gather(batch["input_ids"]).cpu().numpy()
                #     decoded_contexts = tokenizer.batch_decode(contexts, skip_special_tokens=True)
                #     decoded_tokens_contrast = tokenizer.batch_decode(generated_tokens_contrast, skip_special_tokens=True)
                #     # print('+++'+decoded_contexts[0])
                #     # print('===LAB:\t', decoded_labels[0])
                #     # print('===PRED:\t', decoded_preds[0])
                #     # print('===PRED_CONTRAST:\t', decoded_tokens_contrast[0])
                #     # input()
                #
                decoded_source = tokenizer.batch_decode(batch["input_ids"],skip_special_tokens=True)    
                for source,true,pred in zip(decoded_source,decoded_labels,decoded_preds):
                            sample={"source":source,"true":true,"pred":pred}
                            to_dump.append(sample)

                all_cands += decoded_preds
                all_refs += decoded_labels
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    with open(args.output_dir+"prediction_{}.json".format(args.test_file.split("/")[-1].split(".")[0]),"w") as f:
                json.dump(to_dump, f,indent=4)
                    
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    logger.info(f'The final performance on the test set is {result}')

                
                
    if accelerator.is_main_process:
        P, R, F1 = score(all_cands[:len(test_dataset)], all_refs[:len(test_dataset)], lang='en', verbose=True)
        logger.info(f'The final performance on the test set is BERT Score precision:{P.mean()}, recall:{R.mean()}, Fscore:{F1.mean()}')
        with open('./logs.txt', 'a') as f:
            f.write(f'{args.model_name_or_path} {best_epoch} {args.test_file}, result {result}, bertscore{F1.mean()}\n')
    
    #remove everything in this directory
    # import shutil
    # for i in range(args.num_train_epochs):
    #     file_path=args.output_dir+str(i)
    #     if os.path.isdir(file_path):
    #         shutil.rmtree(file_path)
    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(args.output_dir)
    #         if args.push_to_hub:
    #             repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()

# # Data preprocessing

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import datasets
import pandas as pd
import random
random.seed(22)

from datasets import Dataset
from tqdm import tqdm
from underthesea import sent_tokenize as sent_tokenize_uts, word_tokenize as word_tokenize_uts

def sent_tokenize(doc):
    return sent_tokenize_uts(doc)

def word_tokenize(text, format='list'):
    return word_tokenize_uts(text, format=format)


def load_data_as_df(dir):
    res = {
      'name': [],
      'title': [],
      'summary': [],
      'body': [],
      'img_caption': [],
    }
    for file_name in tqdm(os.listdir(dir)):
        file = os.path.join(dir, file_name)
        try:
            with open(file, encoding='utf-8') as f:
                document = f.read().rstrip().split("\n\n")
            for i in range(len(document)):
                document[i] = document[i].replace('\n', ' ')
                
            if len(document) == 3:
                title, summary, doc = document 
                img_caption = ''
            elif len(document) == 4:
                title, summary, doc, img_caption = document 
            res['name'].append(file_name)
            res['title'] += [title]
            res['summary'] += [summary]
            res['body'] += [doc]
            res['img_caption'] += [img_caption]
        except:
            continue
      
    return pd.DataFrame.from_dict(res)

valid_data_df = load_data_as_df('../data/val_tokenized')
train_data_df = load_data_as_df('../data/train_tokenized')

valid_data = Dataset.from_pandas(valid_data_df.sample(n=5000, random_state=22))
train_data = Dataset.from_pandas(train_data_df)


# %%
from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained('Zayt/viRoberta-l6-h384-word-cased')


# %%
encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch, tokenizer=tokenizer):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["body"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch


# %%
# batch_size = 16
batch_size=8

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["name", 'title', 'summary', 'body', 'img_caption']
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

valid_data = valid_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["name", 'title', 'summary', 'body', 'img_caption']
)
valid_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# %% [markdown]
# # Model

# %%
from transformers import EncoderDecoderModel


# %%
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("Zayt/viRoberta-l6-h384-word-cased", "Zayt/viRoberta-l6-h384-word-cased")


# %%
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 130
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4

# %% [markdown]
# # Training

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    fp16=True, 
    output_dir="./save/checkpoint",

    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    num_train_epochs = 6,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 8,
    learning_rate = 5e-5,
    weight_decay=0.01,
    dataloader_num_workers = 4,
# device = 'cuda:1',
    do_train = True,
    do_eval = True,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    warmup_ratio = 0.1,
    seed = 22,
    log_level='info',
    logging_strategy = "steps",
    logging_steps = 150,
    save_total_limit = 4,
    load_best_model_at_end=False,
)

rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2", "rouge1", "rougeL"])

    return {
        "rouge1_fmeasure": round(rouge_output['rouge1'].mid.fmeasure, 4),
        "rouge2_fmeasure": round(rouge_output['rouge2'].mid.fmeasure, 4),
        "rougeL_fmeasure": round(rouge_output['rougeL'].mid.fmeasure, 4),
    }


# %%
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=valid_data,
)
trainer.train()



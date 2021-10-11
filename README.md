#  Small roberta model for Vietnamese with hugginface training scripts
 - 6 layers, 12 attention heads and 384 hidden with whole word masking
 - Pre-tokenize using [underthesea word_tokenize](https://github.com/undertheseanlp/underthesea)
 - Using [Vietnews](https://github.com/ThanhChinhBK/vietnews) train dataset  and first 3gb of [vi Oscar-corpus](https://oscar-corpus.com/post/oscar-2019/).
 - Pretrained model is now available on [huggingface](https://huggingface.co/Zayt/viRoberta-l6-h384-word-cased)

## Fine-tune model for Abstractive summarization task on Vietnews dataset (see abstractive-summarization)
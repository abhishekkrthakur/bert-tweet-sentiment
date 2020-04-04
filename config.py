import tokenizers

MAX_LEN = 160
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 15
BERT_PATH = "/home/abhishek/workspace/bert_base_uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/train_folds.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f"{BERT_PATH}/vocab.txt", 
    lowercase=True
)

import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from tqdm import tqdm
import torch

class CustomTokenizerTrainer:
    def __init__(self, vocab_size, min_frequency=2):
        self.tokenizer = Tokenizer(models.BPE(unk_token="|<unk>|"))
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def train(self, dataset):
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=['|<unk>|', '|<endoftext>|'],
            min_frequency=self.min_frequency
        )

        self.tokenizer.train_from_iterator(dataset, trainer)

        # Add special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<|endoftext|> $A",
            special_tokens=[
                ("<|endoftext|>", self.tokenizer.token_to_id("|<endoftext>|"))
            ]
        )

        self.tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
        
    def save_json(self, path):
        # Ensure the checkpoints folder exists
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", path)

        self.tokenizer.save(path)
        print(f"Tokenizer saved to: {path}")
        print(f"Vocab size: {self.tokenizer.get_vocab_size()} tokens")

    def tokenize_and_save(self, dataset, output_path):
        # Ensure the checkpoints folder exists
        os.makedirs("checkpoints", exist_ok=True)
        output_path = os.path.join("checkpoints", output_path)

        tokenized_samples = []
        for item in tqdm(dataset, desc="Tokenizing Dataset"):
            input_ids = self.tokenizer.encode(item["text"]).ids
            tokenized_samples += input_ids

        # Save tokens as a PyTorch file
        torch.save(torch.LongTensor(tokenized_samples), output_path)  # LongTensor save as 64int
        print(f"Tokenized samples saved to: {output_path}")

if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories")
    # text_data = dataset["train"]["text"]

    # trainer = CustomTokenizerTrainer(vocab_size=10000)
    # trainer.train(text_data)

    # trainer.save_json("bpe_tokenizer_train_10k.json")
    # trainer.tokenize_and_save(dataset["train"], 'tokenized_train_samples_vocab_10k.pt')
    ####validation
    text_data = dataset["validation"]["text"]

    trainer = CustomTokenizerTrainer(vocab_size=10000)
    trainer.train(text_data)

    trainer.save_json("bpe_tokenizer_valid_10k.json")
    trainer.tokenize_and_save(dataset["validation"], 'tokenized_valid_samples_vocab_10k.pt')

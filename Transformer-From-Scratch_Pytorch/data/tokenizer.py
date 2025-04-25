from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders

class CustomTokenizerTrainer:
    def __init__(self, vocab_size, min_frequency = 2):
        self.tokenizer = Tokenizer(models.BPE(unk_token="|<unk>|"))
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
    def train(self, dataset):
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space= False)
        
        trainer  = trainers.BpeTrainer(
            vocab_size = self.vocab_size,
            special_tokens = ['|<unk>|', '|<endoftext>|'],
            min_frequency = self.min_frequency
        )
        
        self.tokenizer.train_from_iterator(dataset , trainer)
        
        # Add special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<|endoftext|> $A",
            special_tokens=[
                ("<|endoftext|>", self.tokenizer.token_to_id("|<endoftext>|"))
            ]
        )


        self.tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
        
    
    def save(self, path):
        self.tokenizer.save(path)
        print(f" Tokenizer saved to: {path}")
        print(f" Vocab size: {self.tokenizer.get_vocab_size()} tokens")


if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories")
    text_data = dataset["validation"]["text"]

    trainer = CustomTokenizerTrainer(vocab_size=10000)
    trainer.train(text_data)

    trainer.save("bpe_tokenizer_medical.json")

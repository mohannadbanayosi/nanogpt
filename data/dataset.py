class Dataset():
    def __init__(self, input_data_path: str = "data/input.txt") -> None:
        with open(input_data_path, "r", encoding="utf-8") as file:
            self.text = file.read()

        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)

        # TODO: check google/sentencepiece and openai/tiktoken
        char_to_index = { char:index for index,char in enumerate(chars) }
        index_to_char = { index:char for index,char in enumerate(chars) }
        self.encode = lambda text: [char_to_index[char] for char in text]
        self.decode = lambda indices: "".join([index_to_char[index] for index in indices])

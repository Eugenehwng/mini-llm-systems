from pathlib import Path
from data import CharTokenizer

DATA_PATH = Path(__file__).parent.parent / 'data' / 'input.txt'
data = DATA_PATH.read_text(encoding='utf-8')

# Sanity check
# print(data[:1000])
# tokenizer = CharTokenizer(data)
# print(tokenizer.vocab_size)
# print(tokenizer.encode('hello'))
# print(tokenizer.stoi)


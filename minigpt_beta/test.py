from pathlib import Path
from data import CharTokenizer

DATA_PATH = Path(__file__).parent.parent / 'data' / 'input.txt'
data = DATA_PATH.read_text(encoding='utf-8')



def test_char_tokenizer():
    tokenizer = CharTokenizer(data)
    assert tokenizer.vocab_size == len(set(data))
    
    enc = tokenizer.encode('hello')
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)


if __name__ == '__main__':
    test_char_tokenizer()
    
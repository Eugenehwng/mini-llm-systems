import os
import requests


# if input.txt doesn't exist in the same directory, get it from data_url
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
        
with open(input_file_path, 'r') as f:
    data = f.read()
    
print(f"length of dataset in characters: {len(data)}")
print(data[:100]) 


    

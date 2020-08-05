import requests
import zipfile
import os
import sys
from tqdm import tqdm
sys.path.insert(1,'./')

url = 'http://dde.binghamton.edu/download/alaska2/abba_weights.zip'
print('Downloading:', url)
file_name = 'abba_weights.zip'
r = requests.get(url)
total_size_in_bytes= int(r.headers.get('content-length', 0))
chunk_size=2**10
with open(file_name, 'wb') as f:
    for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=total_size_in_bytes//chunk_size+1, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'): 
        if chunk:
            f.write(chunk)
            
            
with zipfile.ZipFile(file_name, 'r') as zipref:
    zipref.extractall('./')
    
os.remove(file_name)
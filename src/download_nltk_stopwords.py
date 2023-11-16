from os.path import dirname, exists
from os import makedirs, remove
import nltk

SOURCE = dirname(dirname(__file__))
NLTK_DATA_DIR = f'{SOURCE}/src/nltk_data'

def download_nltk_stopwords():
    
    print('\nVerifying if stopwords resources already exists.')
    if not exists(NLTK_DATA_DIR):
        makedirs(NLTK_DATA_DIR)

        print('Downloading the stopwords resources')

        nltk.download('stopwords', download_dir = NLTK_DATA_DIR, quiet=True)

        print('Download finished.')

        remove(f'{NLTK_DATA_DIR}/corpora/stopwords.zip')
    
    else:
        print('Stopwords resources already exists.')
    
    nltk.data.path.append(NLTK_DATA_DIR)
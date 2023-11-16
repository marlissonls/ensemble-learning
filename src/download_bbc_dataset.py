from urllib  import request as dowload_file
from os.path import dirname, exists
from os import makedirs, remove
from zipfile import ZipFile

SOURCE = dirname(dirname(__file__))
DATASET_DIRECTORY = f'{SOURCE}/src/data/'
BBC_DIRECTORY = f'{DATASET_DIRECTORY}/bbc/'

def download_bbc_news() -> str:
    ''' Download BBC News Dataset and returns the BBC News directory. '''

    if not exists(BBC_DIRECTORY):

        if not exists(DATASET_DIRECTORY):
            makedirs(DATASET_DIRECTORY)

        zipfile_name = 'bbc_dataset.zip'
        zipfile_path = f'{DATASET_DIRECTORY}/{zipfile_name}'

        print('Downloading dataset from BBC...')
        url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
        dowload_file.urlretrieve(url, zipfile_path)

        with ZipFile(zipfile_path, 'r') as zip:
            print('Extracting files from zip...')
            zip.extractall(DATASET_DIRECTORY)
            print('Data extracted succesfully!')

        remove(zipfile_path)

        return BBC_DIRECTORY
    else:
        print('Data already downloaded.')
        return BBC_DIRECTORY

if __name__ == '__main__':
    download_bbc_news()
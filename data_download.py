import yaml
import sys
import requests
import os
import re
import tarfile
import shutil


def download(url, filename, cookies=None):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True, cookies=cookies)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                completed = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format(
                    '█' * completed, '.' * (50-completed)))
                sys.stdout.flush()
    sys.stdout.write('\n')


drive_request = requests.get(
    'https://drive.google.com/uc?export=download&confirm=CONFIRM&id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo')
confirm_page = drive_request.text
confirmation_code = re.findall('confirm=(.{4})', confirm_page)[0]

print('[*] Downloading data...')
download('https://drive.google.com/uc?export=download&confirm=CONFIRM&id=1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo'.replace(
    'CONFIRM', confirmation_code), 'data/viton_resize.tar.gz', cookies=drive_request.cookies)

tarfile.open("data/viton_resize.tar.gz").extractall(path='data/')

shutil.move('data/viton_resize/test/', 'data/test/')
shutil.move('data/viton_resize/train/', 'data/train/')

os.rmdir('data/viton_resize/')
os.remove('data/viton_resize.tar.gz')

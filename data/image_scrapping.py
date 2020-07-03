import os
import io
import hashlib
import requests
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import multiprocessing


def scrapArt(album):
    try:
        r = requests.get(album[1])
        if r.status_code != 200:
            pass
        img = io.BytesIO(r.content)
        img = Image.open(img).convert('RGB')
        if not os.path.isdir(f'{args.save}/{album[0]}'):
            os.mkdir(f'{args.save}/{album[0]}')
        with open(f'{args.save}/{album[0]}/{hashlib.sha1(r.content).hexdigest()[:10]}.jpg', 'wb') as f:
            img.save(f, 'JPEG', quality=100)
    except:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Scrapping Album Art Data for different Genres')
    parser.add_argument('--data', type=str, default='album_image.csv')
    parser.add_argument('--save', type=str, default='images')
    args = parser.parse_args()

    os.mkdir(args.save)

    data = pd.read_csv(args.data)
    data = data.loc[:, ['genre', 'image_url']].to_numpy()
    data = data.tolist()

    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=os.cpu_count())
    outputs = pool.map(scrapArt, data)

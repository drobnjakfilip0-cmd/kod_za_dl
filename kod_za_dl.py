
from duckduckgo_search import DDGS  # Uvoz klase koja omogućava pretragu preko DuckDuckGo
from fastcore.all import *           # fastcore je biblioteka koja daje korisne helper funkcije, npr. L()

def search_images(keywords, max_images=200):
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

urls = search_images('bird photos', max_images=1)
urls[0]

from fastdownload import download_url
dest = 'bird.jpg'
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import *
im = Image.open(dest)

im.to_thumb(256,256)

download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)

from pathlib import Path
import time, warnings
from duckduckgo_search import DDGS
from fastdownload import download_url
from fastai.vision.all import download_images, resize_images

warnings.filterwarnings("ignore")  # sakrij warning-e

searches = ['forest', 'bird']
path = Path('bird_or_not')

for o in searches:
    dest = path/o
    dest.mkdir(exist_ok=True, parents=True)

    try:
        # smanjen broj slika po keyword-u da se izbegne rate limit
        urls_for_keyword = L(DDGS().images(f'{o} photo', max_results=5)).itemgot('image')
        time.sleep(2)  # pauza pre preuzimanja
    except Exception as e:
        print(f"Preskacem {o} zbog greške: {e}")
        continue

    # download i resize bez progress bar-a
    download_images(dest, urls=urls_for_keyword, max_pics=len(urls_for_keyword))
    resize_images(dest, max_size=400, dest=dest, show_progress=False)

    time.sleep(5)  # pauza između keyword-a

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

from fastai.vision.all import get_image_files

files = get_image_files(path)
print(files)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=4)

dls.show_batch(max_n=6)

from fastai.vision.all import *

# learner bez callback-ova i sa malim model_dir
learn = vision_learner(dls, resnet18, metrics=error_rate, cbs=[], model_dir='/tmp/model')

# treniraj direktno bez progress callback-a
for epoch in range(3):
    learn.fit(1)

is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
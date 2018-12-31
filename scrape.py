from jikanpy import Jikan
import logging
import re
import requests
import shutil
import time


logging.basicConfig(filename='logs/scrape.log', level=logging.DEBUG)

anime_id = 21

jikan = Jikan()

characters_staff = jikan.anime(anime_id, extension='characters_staff')

chars = characters_staff['characters']

person_ids = set()
for c in chars:
    va = c['voice_actors']
    for v in va:
        if v['language'] == 'Japanese':
            person_ids.add(v['mal_id'])


for pid in person_ids:
    time.sleep(3)
    person = jikan.person(pid)
    name = re.sub('\s', '', person['name'])
    path = "tmp/%s" % name
    url = person['image_url']
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

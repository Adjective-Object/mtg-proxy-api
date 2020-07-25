# mtg-proxy-api

An API that will generate easy MTG proxies for you.

```sh
# setup and dev
pipenv install
FLASK_ENV=development pipenv run python ./app/__main__.py
```

get mana symbols
On https://mtg.gamepedia.com/Category:Mana_symbols, run

```js
Array.from(document.querySelectorAll("img")).map((x) => x.src);
```

Then take out the image urls and run:

```sh
wget #... urls go here
# convert svg to small pngs
mogrify  -background none -format png -resize 35x35 assets/mana/*.svg
```

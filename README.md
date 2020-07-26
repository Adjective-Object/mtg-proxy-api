# mtg-proxy-api

## Setup

An API that will generate easy MTG proxies for you.

```sh
# setup and dev
pipenv install
DEBUG=true pipenv run python ./app/__main__.py
```

## Asset Sources

Assets are light edits of outputs from [mtg-cardframes](https://github.com/Adjective-Object/mtg-cardframes)

## Updating Mana Symbol Assets

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

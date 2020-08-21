#!/usr/bin/python
# -*- coding: utf-8 -*-

from sanic import Sanic, response
from PIL import Image, ImageFont, ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from itertools import chain, zip_longest
from aiohttp import ClientSession
import asyncio
from hashlib import sha512
import numpy as np
import io
import os
import re

# import matplotlib.pyplot as plt
# plt.imshow(color_mask)
# plt.show()


def load_images_dir(basedir):
    images = dict()
    for path in os.listdir(basedir):
        if path.endswith(".png"):
            base, ext = os.path.splitext(path)
            images[base.upper()] = np.array(Image.open(os.path.join(basedir, path)))

    return images


app = Sanic(__name__)
frame = np.array(Image.open("./assets/frame.png"))
powerbox = np.array(Image.open("./assets/powerbox.png"))
color_mask = np.array(Image.open("./assets/color_mask.png"))[:, :, 0] == 255
color_mask_no_power = (
    np.array(Image.open("./assets/color_mask_no_power.png"))[:, :, 0] == 255
)
title_font = ImageFont.truetype("./assets/TitleFont.ttf", 70)
body_font = ImageFont.truetype("./assets/BodyFont.ttf", 30)
body_font_small = ImageFont.truetype("./assets/BodyFont.ttf", 25)
mana_symbols_dict = load_images_dir("./assets/mana")
rarity_dict = load_images_dir("./assets/setsymbol")
MANA_SYMBOL_SIZE = mana_symbols_dict[next(iter(mana_symbols_dict))].shape[0]
MANA_SYMBOL_PADDING = 6

# The maximum length of the render surface for the title/typeline/powerbox
MAX_RENDERED_TITLE_W = 1200
# Location where images are stored on local disk
IMG_DISK_CACHE = "./img_cache"


def symbol_size(font):
    if font is body_font:
        return 27
    elif font is body_font_small:
        return 21
    else:
        return 12


def symbol_padding(font):
    if font is body_font:
        return 6
    elif font is body_font_small:
        return 5
    else:
        return 2


def pop_symbol(possible_mana_str):
    if len(possible_mana_str) < 3:
        return (None, possible_mana_str)

    if possible_mana_str[0] == "{":
        end_idx = possible_mana_str.find("}")
        if end_idx == -1:
            return (None, possible_mana_str)

        return (possible_mana_str[1:end_idx], possible_mana_str[end_idx + 1 :])

    return (None, possible_mana_str)


def lookup_rarity_image(symbol_name):
    lookup_key = symbol_name.upper()[0]
    rarity_img = (
        rarity_dict[lookup_key] if lookup_key in rarity_dict else rarity_dict["UNKNOWN"]
    )
    return rarity_img


def lookup_mana_image(symbol_name):
    lookup_key = symbol_name.upper().replace("/", "")
    symbol_img = (
        mana_symbols_dict[lookup_key]
        if lookup_key in mana_symbols_dict
        else mana_symbols_dict["UNKNOWN"]
    )
    return symbol_img


def render_mana_symbols(mana_symbol_string):
    symbols = []
    symbol, mana_symbol_string = pop_symbol(mana_symbol_string)
    while symbol:
        symbols.append(symbol)
        symbol, mana_symbol_string = pop_symbol(mana_symbol_string)

    mana_texture = np.zeros(
        (MANA_SYMBOL_SIZE, len(symbols) * (MANA_SYMBOL_SIZE + MANA_SYMBOL_PADDING), 4,),
        dtype=int,
    )
    for i, symbol_name in enumerate(symbols):
        x = i * (MANA_SYMBOL_SIZE + MANA_SYMBOL_PADDING)
        symbol_img = lookup_mana_image(symbol_name)
        mana_texture[0:, x : x + MANA_SYMBOL_SIZE, :] = symbol_img[
            0:MANA_SYMBOL_SIZE, 0:MANA_SYMBOL_SIZE, :
        ]

    return mana_texture


SYMBOL_REGEX = r"\{[^\s,.\{\}]+\}"


def get_text_width(font, line_text, symbol_w):
    occurrences = re.findall(SYMBOL_REGEX, line_text)
    if len(occurrences):
        line_text = re.sub(SYMBOL_REGEX, "", line_text)
    return font.getsize(line_text)[0] + len(occurrences) * symbol_w


def split_lines_for_font(font, text, max_width):
    words = [x for x in text.replace("\n", "\n ").split(" ") if len(x)]
    lines = []
    current_line = words[0]
    for word in words[1:]:
        # forced newlines
        if word == "\n" or current_line[-1] == "\n":
            lines.append(current_line)
            current_line = ""

        next_line = current_line + " " + word if len(current_line) != 0 else word
        next_line_width = get_text_width(
            font, next_line, symbol_size(font) + symbol_padding(font) * 2
        )

        if next_line_width > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = next_line

    if len(current_line) > 0:
        lines.append(current_line)

    return [line.replace("\n", "") for line in lines]


def intercalate(a, b):
    return chain.from_iterable(zip_longest(b, a))


def render_body_text(image_arr, text, x, y, max_width, max_height):
    font = body_font
    line_height = 30
    lines = split_lines_for_font(font, text, max_width)

    # if it would be too large, use the smaller
    # font instead.
    if line_height * len(lines) > max_height:
        font = body_font_small
        line_height = 25
        lines = split_lines_for_font(font, text, max_width)

    rendered_text = Image.new(
        "RGBA", (max_width, line_height * len(lines)), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(rendered_text)

    initial_y = 0
    initial_x = 0
    # center text y if there is only one line
    if len(lines) == 1:
        initial_y = int(max_height / 2 - line_height / 2)
        # center x
        line = lines[0]
        symbols = re.findall(SYMBOL_REGEX, line)
        sublines = re.split(SYMBOL_REGEX, line)
        symbols_width = len(symbols) * (symbol_size(font) + symbol_padding(font) * 2)
        text_width = sum([font.getsize(subline)[0] for subline in sublines])
        initial_x = (max_width - (text_width + symbols_width)) // 2

    print(initial_y, initial_x)

    for i, line in enumerate(lines):
        line_x = initial_x
        symbols = re.findall(SYMBOL_REGEX, line)
        sublines = re.split(SYMBOL_REGEX, line)
        intercalated_lines = list(enumerate(intercalate(symbols, sublines)))
        for j, subline in intercalated_lines:
            if subline is None or len(subline) == 0:
                continue

            if re.match(SYMBOL_REGEX, subline):
                # draw a symbol
                line_x += symbol_padding(font)
                line_symbol_size = symbol_size(font)

                symbol_y = (
                    y
                    + i * line_height
                    + initial_y
                    + (line_height - line_symbol_size) // 2
                )
                symbol_x = x + line_x

                symbol_img = lookup_mana_image(subline[1:-1])
                symbol_img = np.array(
                    Image.fromarray(symbol_img).resize(
                        (line_symbol_size, line_symbol_size)  # , Image.ANTIALIAS
                    )
                )
                composite_alpha(symbol_img, image_arr, symbol_x, symbol_y)

                line_x += line_symbol_size + symbol_padding(x)
            else:
                # draw text
                draw.text((line_x, i * line_height), subline, (0, 0, 0), font=font)
                if i != len(sublines) - 1:
                    w = font.getsize(subline)[0]
                    line_x += w

    rendered_text_arr = np.array(rendered_text)
    composite_alpha(rendered_text_arr, image_arr, x, y + initial_y)


def prep_body_text(text, name):
    text = text.replace("--", "—")  # em dash
    text = text.replace("{bull}:", "•")  # bullet
    text = text.replace("{bull}", "•")  # bullet
    # replace CARDNAME with the name of the card.
    #
    # For names of form "Proper Name, Descriptor",
    # we use the full name for the first appearance, and
    # only use the proper name for the remainder
    short_name_idx = name.find(",")
    short_name = name if short_name_idx == -1 else name[0:short_name_idx]
    text = text.replace("CARDNAME", name, 1)  # name
    text = text.replace("CARDNAME", short_name)  # name

    return text


def composite_alpha(source, target, x, y):
    (h, w, d) = source.shape
    h = min(target.shape[0] - y, h)
    w = min(target.shape[1] - x, w)
    source = source[:h, :w, :]
    alpha = (source[:, :, 3] / 255.0).reshape((source.shape[0], source.shape[1], 1))
    # if target_h != h:
    #     y += (h - target_h) /
    target[y : y + h, x : x + w, :] = (source * (alpha)) + (
        target[y : y + h, x : x + w, :] * (1 - alpha)
    )


def render_title_font(image_arr, name, x, y, max_w, h, centered=False):
    if max_w + x > image_arr.shape[1]:
        max_w = image_arr.shape[1] - x
    # render @2x and resize down to simulate font antialiasing
    rendered_size = title_font.getsize(name)
    w = min(MAX_RENDERED_TITLE_W, rendered_size[0] // 2)
    target_w = min(w, max_w)
    target_h = int(h * 1.0 * target_w // w)
    if centered:
        x = x - target_w // 2
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), name, (0, 0, 0), font=title_font)
    img = img.resize((target_w, target_h), Image.ANTIALIAS)
    rendered_text = np.array(img)
    # alpha blend the text onto the base image
    composite_alpha(rendered_text, image_arr, x, y)


def tint_image(image, color, frame_mask):
    frame_mask_inverse = frame_mask != True
    if color == "w":
        return image
    elif color == "u":
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.33, 0.59, 0.97]]]
        )
        image[:, :, :3][frame_mask_inverse] = image[:, :, :3][
            frame_mask_inverse
        ] * np.array([[[0.9, 0.9, 1]]])
        return image
    elif color == "b":
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.3, 0.3, 0.3]]]
        )
        image[:, :, :3][frame_mask_inverse] = image[:, :, :3][
            frame_mask_inverse
        ] * np.array([[[0.9, 0.9, 0.9]]])
        return image
    elif color == "r":
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.96, 0.41, 0.35]]]
        )
        image[:, :, :3][frame_mask_inverse] = image[:, :, :3][
            frame_mask_inverse
        ] * np.array([[[1, 0.9, 0.9]]])
        return image
    elif color == "g":
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.43, 0.64, 0.34]]]
        )
        image[:, :, :3][frame_mask_inverse] = image[:, :, :3][
            frame_mask_inverse
        ] * np.array([[[0.9, 1, 0.9]]])
        return image
    elif color == "multi":
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.97, 0.93, 0.6]]]
        )
        image[:, :, :3][frame_mask_inverse] = image[:, :, :3][
            frame_mask_inverse
        ] * np.array([[[1, 1, 0.9]]])
        return image
    elif color == "c":
        colors = rgb_to_hsv(image[:, :, :3])
        colors[:, :, 1] = 0
        colors[frame_mask, 2] *= 0.7
        image[:, :, :3] = hsv_to_rgb(colors)
        return image
    elif color == "l":
        # fallback when there is NO color identity.
        # is this a land??
        image[:, :, :3][frame_mask] = image[:, :, :3][frame_mask] * np.array(
            [[[0.7, 0.6, 0.6]]]
        )
        return image
    return image


async def load_image_url(img_url):
    if not os.path.exists(IMG_DISK_CACHE):
        os.makedirs(IMG_DISK_CACHE)

    hashed_url = sha512(img_url.encode("utf-8")).hexdigest()
    _, ext = os.path.splitext(img_url)

    local_path = os.path.join(IMG_DISK_CACHE, hashed_url + "." + ext)
    if os.path.exists(local_path):
        return Image.open(local_path)

    async with ClientSession() as session:
        response = await session.request(method="GET", url=img_url)
        # write local file in 1kb blocks
        with open(local_path, "wb") as fd:
            async for data in response.content.iter_chunked(1024):
                fd.write(data)

    return Image.open(local_path)


def fill_box(destination, source_img, x, y, w, h):
    im_w, im_h = source_img.size
    ratio = max(w / im_w, h / im_h)
    dest_size = (int(im_w * ratio), int(im_h * ratio))
    off_x = (dest_size[0] - w) // 2
    off_y = (dest_size[1] - h) // 2
    print(im_w, im_h, ratio, off_x, off_y, w, h)
    img = source_img.resize(dest_size, Image.ANTIALIAS)
    destination[y : y + h, x : x + w] = np.array(img)[
        off_y : off_y + h, off_x : off_x + w
    ]


@app.route("/card")
async def card(request):
    color = request.args.get("color", default="l")
    name = request.args.get("name", default="[No Name]")
    img_url = request.args.get("img_url", default=None)
    rarity = request.args.get("rarity", default=None)
    cost = request.args.get("cost", default="")
    typeline = request.args.get("typeline", default="[No Type]")
    body = request.args.get("body", default="")
    power = request.args.get("power", default=None)
    toughness = request.args.get("toughness", default=None)

    # add power box if there is a power / toughness
    generated_image = np.copy(frame)
    frame_and_pinline_mask = color_mask
    if power is not None and toughness is not None:
        powerdims = powerbox.shape
        generated_image[-powerdims[0] :, -powerdims[1] :, -powerdims[2] :] = powerbox
    else:
        frame_and_pinline_mask = np.copy(color_mask)
        color_mask_dims = color_mask_no_power.shape
        frame_and_pinline_mask[
            -color_mask_dims[0] :, -color_mask_dims[1] :
        ] = color_mask_no_power

    # tint the frame
    tint_image(generated_image, color, frame_and_pinline_mask)

    if len(cost):
        # render mana cost and offset title
        mana_texture = render_mana_symbols(cost)
        mana_texture_width = mana_texture.shape[1]
        composite_alpha(
            mana_texture,
            generated_image,
            generated_image.shape[1] - 50 - mana_texture_width,  # x
            60,  # y
        )

        render_title_font(generated_image, name, 60, 65, 620 - mana_texture_width, 50)
    else:
        # render title
        render_title_font(generated_image, name, 60, 65, 620, 50)

    # render typeline
    render_title_font(generated_image, typeline, 60, 600, 620, 50)

    # render p/t box
    if power is not None and toughness is not None:
        power_string = power + "/" + toughness
        render_title_font(
            generated_image, power_string, 640, 945, 500, 50, centered=True
        )

    # render body text
    if body:
        body_box = [58, 655, 628, 300]
        padding = 6
        render_body_text(
            generated_image,
            prep_body_text(body, name),
            body_box[0] + padding,
            body_box[1] + padding,
            body_box[2] - padding * 2,
            body_box[3] - padding * 2,
        )

    # set image url
    if img_url and img_url != "undefined":
        loaded_image = await load_image_url(img_url)
        if loaded_image.mode != "RGBA":
            loaded_image = loaded_image.convert("RGBA")
        fill_box(generated_image, loaded_image, 57, 117, 631, 461)
        print("loaded", loaded_image)

    if rarity is not None:
        rarity_img = lookup_rarity_image(rarity)
        composite_alpha(rarity_img, generated_image, 650, 597)

    # encode the response and add it
    img = Image.fromarray(generated_image, "RGBA")
    pngBuffer = io.BytesIO()
    img.save(pngBuffer, format="PNG")
    pngBuffer.seek(0)

    async def streaming_fn(response):
        await response.write(pngBuffer.getbuffer())

    return response.stream(streaming_fn, content_type="image/png")


if __name__ == "__main__":
    is_debug = "DEBUG" in os.environ and os.environ["DEBUG"] == "true"
    app.run(host="0.0.0.0", port=5000, debug=is_debug)


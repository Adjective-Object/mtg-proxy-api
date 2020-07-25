#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, request, send_file
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import io

# import matplotlib.pyplot as plt
# plt.imshow(color_mask)
# plt.show()

app = Flask(__name__)
frame = np.array(Image.open("./assets/frame.png"))
powerbox = np.array(Image.open("./assets/powerbox.png"))
color_mask = (
    np.array(Image.open("./assets/color_mask.png"))[:, :, 0] == 255
)  # import matplotlib.pyplot as plt
title_font = ImageFont.truetype("./assets/TitleFont.ttf", 60)
body_font = ImageFont.truetype("./assets/BodyFont.ttf", 30)
body_font_italic = ImageFont.truetype("./assets/BodyFont.ttf", 30)

# The maximum length of the render surface for the title/typeline/powerbox
MAX_RENDERED_TITLE_W = 1200


def split_lines_for_font(font, text, max_width):
    words = text.split(" ")
    lines = []
    current_line = words[0]
    for word in words[1:]:
        # forced newlines
        if word == "\n" or current_line[-1] == "\n":
            lines.append(current_line)
            current_line = ""

        next_line = current_line + " " + word if len(current_line) != 0 else word
        next_line_width = font.getsize(next_line)[0]

        if next_line_width > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = next_line

    if len(current_line):
        lines.append(current_line)

    return lines


def render_body_text(image_arr, text, x, y, max_width, max_height):
    font = body_font
    line_height = 30
    lines = split_lines_for_font(body_font, text, max_width)

    rendered_text = Image.new(
        "RGBA", (max_width, line_height * len(lines)), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(rendered_text)
    for i, line in enumerate(lines):
        draw.text((0, line_height * i), line, (0, 0, 0), font=font)

    rendered_text_arr = np.array(rendered_text)
    composite_alpha(rendered_text_arr, image_arr, x, y)


def prep_body_text(text):
    text = text.encode("utf-8")
    text = text.replace("--", "—")  # em dash
    text = text.replace("{bull}", "•")  # bullet

    return text


def composite_alpha(source, target, x, y):
    (h, w, d) = source.shape
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
    w = min(MAX_RENDERED_TITLE_W, rendered_size[0] / 2)
    target_w = min(w, max_w)
    target_h = int(h * 1.0 * target_w / w)
    if centered:
        x = x - target_w / 2
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), name, (0, 0, 0), font=title_font)
    img = img.resize((target_w, target_h), Image.ANTIALIAS)
    rendered_text = np.array(img)
    # alpha blend the text onto the base image
    composite_alpha(rendered_text, image_arr, x, y)


def tint_image(image, color):
    if color == "w":
        return image
    elif color == "u":
        image[:, :, :3][color_mask] = image[:, :, :3][color_mask] * np.array(
            [[[0.37, 0.32, 0.95]]]
        )
        return image
    elif color == "b":
        image[:, :, :3][color_mask] = image[:, :, :3][color_mask] * np.array(
            [[[0.65, 0.13, 0.65]]]
        )
        return image
    elif color == "r":
        image[:, :, :3][color_mask] = image[:, :, :3][color_mask] * np.array(
            [[[0.96, 0.41, 0.35]]]
        )
        return image
    elif color == "g":
        image[:, :, :3][color_mask] = image[:, :, :3][color_mask] * np.array(
            [[[0.11, 0.60, 0.05]]]
        )
        return image
    elif color == "c":
        colors = rgb_to_hsv(image[:, :, :3])
        colors[:, :, 1] = 0
        image[:, :, :3][color_mask] = hsv_to_rgb[colors]
        return image
    return image


@app.route("/card")
def hello():
    color = request.args.get("color", default="w")
    name = request.args.get("name", default="[No Name]")
    img_url = request.args.get("img_url", default=None)
    cost = request.args.get("cost", default="")
    typeline = request.args.get("typeline", default="[No Type]")
    body = request.args.get("body", default="")
    power = request.args.get("power", default=None)
    toughness = request.args.get("toughness", default=None)

    # add power box if there is a power / toughness
    generated_image = np.copy(frame)
    if power is not None and toughness is not None:
        powerdims = powerbox.shape
        generated_image[-powerdims[0] :, -powerdims[1] :, -powerdims[2] :] = powerbox

    # tint the frame
    tint_image(generated_image, color)

    # render title
    render_title_font(generated_image, name, 60, 68, 600, 50)

    # render typeline
    render_title_font(generated_image, typeline, 60, 600, 600, 50)

    # render p/t box
    if power is not None and toughness is not None:
        power_string = power + "/" + toughness
        render_title_font(
            generated_image, power_string, 640, 945, 500, 50, centered=True
        )

    # render body text
    if body:
        render_body_text(generated_image, prep_body_text(body), 58, 655, 628, 300)

    # encode the response and add it
    img = Image.fromarray(generated_image, "RGBA")
    pngBuffer = io.BytesIO()
    img.save(pngBuffer, format="PNG")
    pngBuffer.seek(0)
    return send_file(pngBuffer, mimetype="image/png",)


if __name__ == "__main__":
    app.run()


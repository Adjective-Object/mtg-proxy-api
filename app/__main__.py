from flask import Flask, request, send_file
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import io

import matplotlib.pyplot as plt

app = Flask(__name__)
frame = np.array(Image.open("./assets/frame.png"))
powerbox = np.array(Image.open("./assets/powerbox.png"))
color_mask = (
    np.array(Image.open("./assets/color_mask.png"))[:, :, 0] == 255
)  # import matplotlib.pyplot as plt
title_font = ImageFont.truetype("./assets/TitleFont.ttf", 60)

# plt.imshow(color_mask)
# plt.show()


MAX_RENDERED_W = 1200


def render_title(image_arr, name, x, y, max_w, h):
    if max_w + x > image_arr.shape[1]:
        max_w = image_arr.shape[1] - x
    # render @2x and resize down to simulate font antialiasing
    rendered_size = title_font.getsize(name)
    w = min(MAX_RENDERED_W, rendered_size[0] / 2)
    target_w = min(w, max_w)
    target_h = int(h * 1.0 * target_w / w)
    print(w, target_w, target_h)
    img = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), name, (0, 0, 0), font=title_font)
    img = img.resize((target_w, target_h), Image.ANTIALIAS)
    print(img.width, img.height, ":", target_w, target_h)
    rendered_text = np.array(img)
    # alpha blend the text onto the base image
    alpha = (rendered_text[:, :, 3] / 255.0).reshape(
        (rendered_text.shape[0], rendered_text.shape[1], 1)
    )
    # if target_h != h:
    #     y += (h - target_h) /
    image_arr[y : y + target_h, x : x + target_w, :] = (rendered_text * (alpha)) + (
        image_arr[y : y + target_h, x : x + target_w, :] * (1 - alpha)
    )


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
    if power is not None and color is not None:
        powerdims = powerbox.shape
        generated_image[-powerdims[0] :, -powerdims[1] :, -powerdims[2] :] = powerbox

    # tint the frame
    tint_image(generated_image, color)

    # render title
    render_title(generated_image, name, 60, 68, 600, 100)

    # encode the response and add it
    img = Image.fromarray(generated_image, "RGBA")
    pngBuffer = io.BytesIO()
    img.save(pngBuffer, format="PNG")
    pngBuffer.seek(0)
    return send_file(pngBuffer, mimetype="image/png",)


if __name__ == "__main__":
    app.run()


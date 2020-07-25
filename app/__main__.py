from flask import Flask, request, send_file
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

# import matplotlib.pyplot as plt
import io

app = Flask(__name__)
frame = np.array(Image.open("./assets/frame.png"))
powerbox = np.array(Image.open("./assets/powerbox.png"))
color_mask = np.array(Image.open("./assets/color_mask.png"))[:, :, 0] == 255
# plt.imshow(color_mask)
# plt.show()


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

    # encode the response and add it
    img = Image.fromarray(generated_image, "RGBA")
    pngBuffer = io.BytesIO()
    img.save(pngBuffer, format="PNG")
    pngBuffer.seek(0)
    return send_file(pngBuffer, mimetype="image/png",)


if __name__ == "__main__":
    app.run()


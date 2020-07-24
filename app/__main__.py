from flask import Flask, request, send_file
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
frame = np.array(Image.open("./assets/frame.png"))
powerbox = np.array(Image.open("./assets/powerbox.png"))


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

    # encode the response and add it
    img = Image.fromarray(generated_image, "RGBA")
    pngBuffer = io.BytesIO()
    img.save(pngBuffer, format="PNG")
    pngBuffer.seek(0)
    return send_file(pngBuffer, mimetype="image/png",)


if __name__ == "__main__":
    app.run()


from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from ice_breaker import ice_break_with

load_dotenv()

app = Flask(__name__)


# default route
@app.route("/")
def index():
    # render html template.
    return render_template("index.html")


# /process route.
@app.route("/process", methods=["POST"])
def process():
    # gets the name input from user.
    name = request.form["name"]
    # gets the tuple[pydanticObj , url]
    summary, profile_pic_url = ice_break_with(name=name)
    #converts to json for usage in frontend.
    return jsonify(
        {
            "summary_and_facts": summary.to_dict(),
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":

    #default run on localhost.
    app.run(host="0.0.0.0", debug=True)
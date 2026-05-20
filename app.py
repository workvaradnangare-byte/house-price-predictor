from flask import *
import pandas as pd
from pickle import load

app = Flask(__name__)

# Load trained model
with open("Model.pkl", "rb") as f:
    model = load(f)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        bedrooms = float(request.form.get("bhk"))
        size_sqft = float(request.form.get("size_sqft"))
        bathroom = float(request.form.get("bathroom"))

        furnishing = request.form.get("furnishing_status")

        # One-hot encoding
        if furnishing == "Furnished":
            f1, f2, f3 = 1, 0, 0

        elif furnishing == "Semi-Furnished":
            f1, f2, f3 = 0, 1, 0

        else:
            f1, f2, f3 = 0, 0, 1

        # Create dataframe
        data = pd.DataFrame([[

            bedrooms,
            size_sqft,
            bathroom,
            f1,
            f2,
            f3

        ]], columns=[

            "bhk",
            "size_sqft",
            "bathroom",
            "furnishing_status_Furnished",
            "furnishing_status_Semi-Furnished",
            "furnishing_status_Unfurnished"

        ])

        # Prediction
        prediction = model.predict(data)[0]

        prediction = int(prediction)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
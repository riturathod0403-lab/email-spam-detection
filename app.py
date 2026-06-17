from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    message = request.form["message"]

    transformed = vectorizer.transform([message])

    prediction = model.predict(transformed)[0]

    confidence = model.predict_proba(transformed).max() * 100

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
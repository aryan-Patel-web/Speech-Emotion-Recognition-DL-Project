from flask import Flask, request, render_template
import pickle
import os
from extract_feature import extract_feature

app = Flask(__name__)

# Load model
with open("MLPClassifier_MODEL.pkl", "rb") as f:
    model = pickle.load(f)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["audio"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        os.remove(file_path)  # clean up uploaded file
        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)

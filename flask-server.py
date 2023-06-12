import logging
from flask import Flask, request, jsonify, render_template, abort

from utils.Onnx import OnnxInference
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
onnx = OnnxInference()


@app.route("/", methods=["GET"])
def index():
    if not app.debug:
        abort(404)
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    b64 = data.get('imageBase64')

    result = onnx.predict_base64([b64])
    return jsonify({'status': "DONE", 'result': str(result[0])})


if __name__ == '__main__':
    app.run(debug=True)

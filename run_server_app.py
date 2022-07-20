import dill
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


with open("./models/logreg_pipeline.dill", 'rb') as f:
    model = dill.load(f)


@app.route('/', methods=['GET'])
def general():
    return 'Welcome to toxic comments prediction process'


@app.route('/predict', methods=['POST'])
def predict():
    data = {'succes': False}

    comment_text = ''
    request_json = request.get_json()

    if request_json['comment_text']:
        comment_text = request_json['comment_text']

    preds = model.predict_proba(pd.DataFrame({'comment_text': [comment_text]}))

    data['predictions'] = preds[:, 1][0]
    data['comment_text'] = comment_text
    data['succes'] = True

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5001')
import requests
import urllib.request
import json
import pandas as pd
import numpy as np


def send_json(x):
    comment_text = x
    body = {'comment_text': comment_text}
    my_url = 'http://127.0.0.1:5001' + '/predict'
    headers = {'content-type': 'application/json; charset=utf-8'}
    response = requests.post(my_url, json=body, headers=headers)

    return response.json()


if __name__ == '__main__':
    comment = str(input('Введите комментарий для проверки '))
    pred = np.round(send_json(comment)['predictions'] * 100, 1)
    print(f'Ваш комментарий токсичен на {pred}% !')

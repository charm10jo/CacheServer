import unittest
import pandas as pd
import pickle
import requests
from flask import Flask, request
import json
from tqdm import tqdm
import time
import unicodedata
import pickle

uri = "http://107.23.108.110:5000/" # 수정필요
tfitf_table_path = './tfidf.csv'

df = pd.read_csv(tfitf_table_path)


def get_division(text):
    start = time.time()
    # WAS로부터 전처리된 데이터를 받아옵니다.
    res = requests.post(uri+'tokenize', data=json.dumps({"text":text}), headers={'Content-Type': 'text/text; charset=utf-8'})
    tokenized = res.json()
    arr_divisions = []

    # nouns 는 전처리된 데이터 중 명사만을 추출한 것 입니다.
    nouns = tokenized["nouns"]

    # talk 는 전처리된 데이터 자체로, 한국어로 된 문장에 가깝습니다.
    talk = tokenized["tok_symptoms"]

    for i in range(14):
        arr_divisions.append(0)

    # get nouns' length
    tokens_len = len(nouns)

    # 만약 nouns의 길이가 0이라면, 에러 메시지를 보냅니다. 
    if(tokens_len == 0):
        return -1

    not_included = []

    # 캐시 서버 내부 단어장(이하 TF-ITF 테이블)을 탐색, 단어장에 없는 단어 갯수를 셉니다.
    for val in nouns:
        if val not in df.columns:
            not_included.append(val)

    included = [x for x in nouns if x not in not_included]

    # 캐시 서버에 있는 단어라면, 그 단어의 분과별 가중치 정보를 가져옵니다. 단어들의 가중치를 분과 방향으로 합합니다.
    # 그 결과, 14x1 크기의 테이블을 얻을 수 있습니다. 
    weights = df[included]
    weights_of_divisions = (weights.sum(axis=1).values.tolist())

    # 가중치 합 테이블의 표준편차를 구합니다. 이때 표준편차의 분모는 TF-IDF 테이블에 없는 단어 갯수도 포합하고 있습니다.
    # 테이블에 없는 단어가 많이 포함된 문장을 입력 받을수록, 표준편차 크기는 작아지고, "비증상(not symptoms)" 카테고리로 빠질 가능성도 높아집니다.  
    avg = sum(weights_of_divisions)/tokens_len
    std = 0

    for val in weights_of_divisions:
        std += ((val - avg) ** 2)
    std = (std ** (1/2)) /tokens_len

    # 가중치 합 테이블에서 가장 높은 가중치 합을 가지는 분과가 캐시 서버의 분과 예측이 됩니다. 
    # 표준편차와 단어장에 없는 단어의 비율(miss rate)이 기준치에 미치지 못한다면, AI 서버로 증상 정보를 보내거나 또는 비증상으로 보고 반환합니다.
    predict_div = weights_of_divisions.index(max(weights_of_divisions))
    missed = len(not_included) / tokens_len

    end = time.time()
    this_time = end - start

    # 캐시 서버의 예측을 신뢰하는 경우입니다.
    if std >= 0.032 and missed < 0.30:
        return {"result":"cached", "division":predict_div, "std": std, "time": this_time}
    # 비증상 정보를 입력받았을 것을 가정하는 경우입니다. 
    elif std < 0.032 and missed >= 0.30:
        return {"result":"wrong", "division": "wrong", "std":std, "missed":missed}
    # AI 서버로 증상 정보를 보내 분과 정보를 반환받습니다. 
    else:
        predict_uri = uri + '/predict'
        res = requests.post(predict_uri, data=json.dumps({"token":talk}), headers={'Content-Type': 'text/text; charset=utf-8'})
        
        res_json = res.json()
        division = res_json["division"]
        probability = float(res_json["prob"])

        # 만약 AI가 한 예측의 신뢰도가 90% 이하라면, chatserver는 사용자에게 상담원 매칭을 제안합니다.
        if(probability <= 90):
            return {"result":"AI result: bad", "division":division, "prob": probability,  "std": std}
        # AI가 한 예측의 신뢰도가 충분히 높다면, 분과 정보를 반환합니다.
        else:
            return {"result":"AI result","std": std,  "division": division, "prob":probability, "time": this_time}
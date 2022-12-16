import pandas as pd
import requests
from flask import Flask, request
import json

was_uri = "http://100.24.235.22:5000/" # 수정필요
ws_uri = "https://charm10jo-skywalker.shop/"
tfitf_table_path = './tfidf.csv'

df = pd.read_csv(tfitf_table_path)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods=['POST'])
def get_division():
    request_json = json.loads(request.get_data(), encoding='utf-8')
       
    symptoms = request_json['symptoms']
    language = request_json['language']
    priority = request_json['priority']
    latitude = request_json['latitude']
    longitude = request_json['longitude']
    
    # WAS로부터 전처리된 증상 데이터를 받아옵니다.
    res = requests.post(was_uri+'tokenize', data=json.dumps({"text":symptoms}), headers={'Content-Type': 'text/text; charset=utf-8'})
    
    tokenized = res.json()
    arr_divisions = []
    
    # nouns 는 전처리된 데이터 중 명사만을 추출한 것 입니다.
    nouns = tokenized["nouns"]
    # talk 는 전처리된 데이터 자체로, 한국어로 된 원 문장에 가깝습니다.
    talk = tokenized["tok_symptoms"]
    for i in range(14):
        arr_divisions.append(0)

    tokens_len = len(nouns)
    
    # 만약 nouns의 길이가 0이라면, 에러 메시지를 보냅니다. 
    if(tokens_len == 0):
        return -1

    not_included = []

    # 캐시 서버 내부 단어장(이하 TF-ITF 테이블)을 탐색, 단어장에 없는 단어 갯수를 셉니다.
    for val in nouns:
        if val not in df.columns:
            not_included.append(val)
    # 원문 중 단아장이 가지고 있는 단어들과 그 가중치를 가져옵니다. 
    included = [x for x in nouns if x not in not_included]

    # 만약 원문이 가지고 있는 어떤 단어도 TF-ITF 테이블 단어장이 가지고 있지 않다면, AI Prediction을 호출합니다. 
    if(len(included) == 0):
        predict_uri = was_uri + '/predict'

        res = requests.post(predict_uri, data=json.dumps({"token":talk}), headers={'Content-Type': 'text/text; charset=utf-8'})

        res_json = res.json()
        division = res_json["division"]
        second_division = (res_json["prob"])
        
        # 예측 1순위 분과와 2순위 분과를 가져와 웹서버를 검색, 알맞은 병원 정보를 가져옵니다. 
        ws_res =  requests.post(ws_uri, data={
                "division": division,
                "language": language,
                "priority": priority,
                "latitude": latitude,
                "longitude": longitude
        })

        ws_res_json = ws_res.json()["result"]
        ws_res2 = requests.post(ws_uri, data ={
                "division": second_division,
                "language": language,
                "priority": priority,
                "latitude": latitude,
                "longitude": longitude    
        })

        ws_res_json2 = ws_res2.json()["result"]
        userList = []

        # 검색 결과 중 중복된 경우를 제외, 결과를 클라이언트에게 반환합니다. 
        for i in range(len(ws_res_json)):
            coe_count = 0
            for j in range(len(ws_res_json2)):
                if(ws_res_json[i]["hospitalName"] == ws_res_json2[j]["hospitalName"]):
                    break
                else:
                    coe_count += 1
            if coe_count == len(ws_res_json2):
                userList.append(ws_res_json[i])


        return {"result": userList + ws_res_json2, "by":"ai"}

    # 캐시 서버에 있는 단어라면, 그 단어의 분과별 가중치 정보를 가져옵니다. 단어들의 가중치를 분과 방향으로 합합니다.
    # 그 결과, 14x1 크기의 테이블을 얻을 수 있습니다. 
    weights = df[included]
    weights_of_divisions = (weights.sum(axis=1).values.tolist())

    # 가중치 합 테이블의 표준편차를 구합니다. 단어의 길이로 나누어주므로, 입력받은 문장의 길이가 길거나 짧은 경우에도 일관적인 결과를 보장할 수 있을 것으로 기대합니다.
    avg = sum(weights_of_divisions)/len(included)
    std = 0

    for val in weights_of_divisions:
        std += ((val - avg) ** 2)
    std = (std ** (1/2)) /tokens_len

    # 가중치 합 테이블에서 가장 높은 가중치 합을 가지는 분과가 캐시 서버의 분과 예측이 됩니다. 
    predict_div = weights_of_divisions.index(max(weights_of_divisions))

    # 캐시 서버의 예측을 신뢰하는 경우입니다.
    if std >= 0.038:
        ws_res =  requests.post(ws_uri, data={
                "division": predict_div, 
                "language": language,
                "priority": priority,
                "latitude": latitude,
                "longitude": longitude
            })
        ws_res_json = ws_res.json()["result"]
        return {"result": ws_res_json, "by":"cache", "div":predict_div}

    # AI 서버로 증상 정보를 보내 분과 정보를 반환받습니다. 
    else:
        predict_uri = was_uri + '/predict'
        
        res = requests.post(predict_uri, data=json.dumps({"token":talk}), headers={'Content-Type': 'text/text; charset=utf-8'})
        res_json = res.json()
        division = res_json["division"]
        second_division = (res_json["prob"])

        # 예측 1순위 분과와 2순위 분과를 가져와 웹서버를 검색, 알맞은 병원 정보를 가져옵니다. 
        ws_res =  requests.post(ws_uri, data={
                "division": division, 
                "language": language,
                "priority": priority,
                "latitude": latitude,
                "longitude": longitude
        })
        
        ws_res_json = ws_res.json()["result"]
        ws_res2 = requests.post(ws_uri, data ={
                "division": second_division,
                "language": language,
                "priority": priority,
                "latitude": latitude,
                "longitude": longitude
        })
        ws_res_json2 = ws_res2.json()["result"]
        userList = []
        
        # 검색 결과 중 중복된 경우를 제외, 결과를 클라이언트에게 반환합니다. 
        for i in range(len(ws_res_json)):
            coe_count = 0
            for j in range(len(ws_res_json2)):
                if(ws_res_json[i]["hospitalName"] == ws_res_json2[j]["hospitalName"]):
                    break
                else:
                    coe_count += 1
            if coe_count == len(ws_res_json2):
                userList.append(ws_res_json[i])
                    
        return {"result": userList + ws_res_json2, "by":"ai"}

if __name__ == '__main__':
   app.run('0.0.0.0',port=5001,debug=True)

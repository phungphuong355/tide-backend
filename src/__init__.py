from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import json

# Flask Server Backend
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def hello():
    return "Hello World!"


@app.route('/model/xgboost1', methods=['GET'])
def predictXgboost1():
    try:
        # validate file
        if 'time' not in json.loads(request.data):
            raise Exception('time is require')
        if 'wlvungtau' not in json.loads(request.data):
            raise Exception("wlvungtau is require")

        time = json.loads(request.data)['time']
        vungtau = json.loads(request.data)['wlvungtau']

        df = pd.DataFrame({
            'Time': time,
            'WL_vungtau': vungtau
        })

        df.loc[:, 'WL_vungtau_lag1'] = df['WL_vungtau'].shift(1)
        df.loc[:, 'WL_vungtau_lag2'] = df['WL_vungtau'].shift(2)
        df.loc[:, 'WL_vungtau_lag3'] = df['WL_vungtau'].shift(3)

        X_pred = df[['WL_vungtau', 'WL_vungtau_lag1',
                     'WL_vungtau_lag2', 'WL_vungtau_lag3']]

        xgboost1 = open(
            './src/model/XGBOOST1.pickle', "rb")
        xgboost1 = pickle.load(xgboost1)

        result = xgboost1.predict(X_pred)

        predict = pd.DataFrame({
            "Time": df['Time'],
            "Predict": result
        })

        fn = predict.to_dict(orient="records")

        res = jsonify({'message': 'ok', "result": fn})
        res.status_code = 200
        return res
    except Exception as error:
        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


@app.route('/model/xgboost2', methods=['GET'])
def predictXgboost2():
    try:
       # validate file
        if 'time' not in json.loads(request.data):
            raise Exception('time is require')
        if 'wlvungtau' not in json.loads(request.data):
            raise Exception("wlvungtau is require")
        if 'wlnhabe' not in json.loads(request.data):
            raise Exception("wlnhabe is require")

        time = json.loads(request.data)['time']
        vungtau = json.loads(request.data)['wlvungtau']
        nhabe = json.loads(request.data)['wlnhabe']

        df = pd.DataFrame({
            'Time': time,
            'WL_vungtau': vungtau,
            'WL_nhabe': nhabe
        })

        df.loc[:, 'WL_vungtau_lag1'] = df['WL_vungtau'].shift(1)
        df.loc[:, 'WL_vungtau_lag2'] = df['WL_vungtau'].shift(2)
        df.loc[:, 'WL_vungtau_lag3'] = df['WL_vungtau'].shift(3)

        df.loc[:, 'WL_nhabe_lag1'] = df['WL_nhabe'].shift(1)
        df.loc[:, 'WL_nhabe_lag2'] = df['WL_nhabe'].shift(2)
        df.loc[:, 'WL_nhabe_lag3'] = df['WL_nhabe'].shift(3)

        X_pred = df[['WL_vungtau', 'WL_nhabe', 'WL_vungtau_lag1', 'WL_vungtau_lag2',
                     'WL_vungtau_lag3', 'WL_nhabe_lag1', 'WL_nhabe_lag2', 'WL_nhabe_lag3']]

        xgboost2 = open(
            './src/model/XGBOOST2.pickle', "rb")
        xgboost2 = pickle.load(xgboost2)

        result = xgboost2.predict(X_pred)

        predict = pd.DataFrame({
            "Time": df['Time'],
            "Predict": result
        })

        fn = predict.to_dict(orient="records")

        res = jsonify({'message': 'ok', 'result': fn})
        res.status_code = 200
        return res
    except Exception as error:
        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


@app.route('/model/xgboost3', methods=['GET'])
def predictXgboost3():
    try:
       # validate file
        if 'time' not in json.loads(request.data):
            raise Exception('time is require')
        if 'wlvungtau' not in json.loads(request.data):
            raise Exception("wlvungtau is require")
        if 'wlnhabe' not in json.loads(request.data):
            raise Exception("wlnhabe is require")
        if 'wlphuan' not in json.loads(request.data):
            raise Exception("wlphuan is require")

        time = json.loads(request.data)['time']
        vungtau = json.loads(request.data)['wlvungtau']
        nhabe = json.loads(request.data)['wlnhabe']
        phuan = json.loads(request.data)['wlphuan']

        df = pd.DataFrame({
            'Time': time,
            'WL_vungtau': vungtau,
            'WL_nhabe': nhabe,
            'WL_phuan': phuan
        })

        df.loc[:, 'WL_vungtau_lag1'] = df['WL_vungtau'].shift(1)
        df.loc[:, 'WL_vungtau_lag2'] = df['WL_vungtau'].shift(2)
        df.loc[:, 'WL_vungtau_lag3'] = df['WL_vungtau'].shift(3)

        df.loc[:, 'WL_nhabe_lag1'] = df['WL_nhabe'].shift(1)
        df.loc[:, 'WL_nhabe_lag2'] = df['WL_nhabe'].shift(2)
        df.loc[:, 'WL_nhabe_lag3'] = df['WL_nhabe'].shift(3)

        df.loc[:, 'WL_phuan_lag1'] = df['WL_phuan'].shift(1)
        df.loc[:, 'WL_phuan_lag2'] = df['WL_phuan'].shift(2)
        df.loc[:, 'WL_phuan_lag3'] = df['WL_phuan'].shift(3)

        X_pred = df[['WL_vungtau', 'WL_nhabe', 'WL_phuan', 'WL_vungtau_lag1', 'WL_vungtau_lag2', 'WL_vungtau_lag3',
                     'WL_nhabe_lag1', 'WL_nhabe_lag2', 'WL_nhabe_lag3', 'WL_phuan_lag1', 'WL_phuan_lag2', 'WL_phuan_lag3']]

        xgboost3 = open(
            './src/model/XGBOOST3.pickle', "rb")
        xgboost3 = pickle.load(xgboost3)

        result = xgboost3.predict(X_pred)

        predict = pd.DataFrame({
            "Time": df['Time'],
            "Predict": result
        })

        fn = predict.to_dict(orient="records")

        res = jsonify({'message': 'ok', 'result': fn})
        res.status_code = 200
        return res
    except Exception as error:
        res = jsonify({'message': 'Bad request', 'content': str(error)})
        res.status_code = 400
        return res


# Start Backend
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='1138')

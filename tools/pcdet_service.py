from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/pointCloud/recognition", methods=['POST'])
def recognition():
    datas = request.json['datas']
    result = {
        "code": "OK",
        "message": "",
        "data": [
            {
                "id": data['id'],
                "code": 0,
                "message": "",
                "objects": [
                    {
                        "label": "TEST_OBJECT",
                        "confidence": 0.99,
                        "x": 1.0,
                        "y": 2.0,
                        "z": 3.0,
                        "dx": 4.0,
                        "dy": 5.0,
                        "dz": 6.0,
                        "rotX": 0.0,
                        "rotY": 0.0,
                        "rotZ": 0.0
                    }
                ]
            }
            for data in datas
        ]
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, jsonify, request

# creating a Flask app
app = Flask(__name__)


# to check the if the API is up
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    if request.method == 'GET':
        data = "API is up!"
        return jsonify({'status': data})


# driver function
if __name__ == '__main__':
    app.run(debug=True)

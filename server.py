from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv
import logging

# key setup
load_dotenv()
KEY = os.getenv("SECRET_KEY")

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# server setup
app = Flask(__name__)
app.config['SECRET_KEY'] = KEY
app.debug = True
socketio = SocketIO(app)

# route setup
@app.route('/')
def index():
    logger.info('Serving root')
    return render_template('index.html')

# socket setup
@socketio.on('event', namespace='/test')
def response(message):
    logger.info('respionding to request')
    emit('server_response',{'data':'hello from server'})

@socketio.on('json')
def handle_json(json):
    send(json, json=True)

@socketio.on('connect', namespace='test')
def connect():
    emit('server_response', {'data': 'Connected to server'})

@socketio.on('disconnect', namespace='/test')
def disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    logger.info('Server Running')
    socketio.run(app, host='0.0.0.0', port=8080, keyfile='key.pem', certfile='cert.pem')

import asyncio
import json
import os
import sys

from deepgram import Deepgram
from flask import Flask, jsonify, request

# Creating a Flask app
app = Flask(__name__)


# To check the if the API is up
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    if request.method == 'GET':
        data = "API is up!"
        return jsonify({'status': data})


@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if request.method == 'POST':
        data = "This is a place holder for output from Deepgram"
        conv_id = request.form["conversation_id"]
        dirname = "./audio_files/"
        save_path = os.path.join(dirname, conv_id + ".mp3")
        request.files['audio_file'].save(save_path)

        try:
            # Transcribe to text using Deepgram
            # TODO: try out speaker diarization
            data = asyncio.run(deepgram_stt(save_path))

            # pass it to NLU

        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print(f'line {line_number}: {exception_type} - {e}')

        return jsonify({'result': data})


async def deepgram_stt(FILE: str):
    # FILE = './audio_files/6min.mp3'

    MIMETYPE = 'audio/mp3'

    # Initialize the Deepgram SDK
    deepgram = Deepgram(os.environ["DEEPGRAM_API_KEY"])

    # Check whether requested file is local or remote, and prepare source
    if FILE.startswith('http'):
        source = {'url': FILE}
    else:
        audio = open(FILE, 'rb')
        source = {'buffer': audio, 'mimetype': MIMETYPE}

    # Send the audio to Deepgram and get the response
    response = await asyncio.create_task(
        deepgram.transcription.prerecorded(
            source,
            {
                'punctuate': True,
                'model': 'nova',
                'diarize': True,
            }
        )
    )

    print(
        f'Speech to text based on audio input: {json.dumps(response, indent=4)}')
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]


# driver function
if __name__ == '__main__':
    app.run(port=5999, debug=True)

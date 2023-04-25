import asyncio
import json
import os
import sys

from deepgram import Deepgram
from flask import Flask, jsonify, request

# creating a Flask app
app = Flask(__name__)


# to check the if the API is up
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
        save_path = os.path.join(dirname, conv_id + ".wav")
        request.files['audio_file'].save(save_path)

        try:
            data = asyncio.run(deepgram(save_path))
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print(f'line {line_number}: {exception_type} - {e}')

        return jsonify({'result': data})


async def deepgram(FILE: str):
    # FILE = './audio_files/convo.mp3'

    # Mimetype for the file you want to transcribe
    # Include this line only if transcribing a local file
    # Example: audio/wav
    MIMETYPE = 'audio/mp3'

    # Initialize the Deepgram SDK
    deepgram = Deepgram(os.environ["DEEPGRAM_API_KEY"])

    # Check whether requested file is local or remote, and prepare source
    if FILE.startswith('http'):
        # file is remote
        # Set the source
        source = {
            'url': FILE
        }
    else:
        # file is local
        # Open the audio file
        audio = open(FILE, 'rb')

        # Set the source
        source = {
            'buffer': audio,
            'mimetype': MIMETYPE
        }

    # Send the audio to Deepgram and get the response
    response = await asyncio.create_task(
        deepgram.transcription.prerecorded(
            source,
            {
                'punctuate': True,
                'model': 'nova',
            }
        )
    )

    # Write the response to the console
    print(json.dumps(response["results"]["channels"][0]["alternatives"][0]["transcript"], indent=4))

    # Write only the transcript to the console
    return response["results"]["channels"][0]["alternatives"][0]["transcript"]


# driver function
if __name__ == '__main__':
    app.run(debug=True)

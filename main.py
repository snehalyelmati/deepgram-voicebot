import asyncio
import json
import os
import sys

from deepgram import Deepgram
from flask import Flask, jsonify, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast, BloomForCausalLM

# Creating a Flask app
app = Flask(__name__)


# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]="0.0"


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
            response = asyncio.run(deepgram_stt(save_path))
            verbatim, summaries = response["results"]["channels"][0]["alternatives"][0]["transcript"], \
                response["results"]["channels"][0]["alternatives"][0]["summaries"]

            return jsonify(
                {
                    'verbatim': verbatim,
                    'summaries': summaries
                }
            )

        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print(f'line {line_number}: {exception_type} - {e}')

        return jsonify({'result': 'Something wrong with Deepgram API'})


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
                'summarize': True,
            }
        )
    )

    print(f'Speech to text based on audio input: {json.dumps(response, indent=4)}')
    # return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    # return response["results"]["channels"][0]["alternatives"][0]["transcript"], \
    #     response["results"]["channels"][0]["alternatives"][0]["summaries"]
    return response


@app.route('/fraud_detection', methods=['POST'])
def fraud_detection():
    if request.method == 'POST':
        conv_id = request.form["conversation_id"]
        dirname = "./audio_files/"
        save_path = os.path.join(dirname, conv_id + ".mp3")
        request.files['audio_file'].save(save_path)

        try:
            # Transcribe to text using Deepgram
            response = asyncio.run(deepgram_stt(save_path))
            print(f'Speech to text based on audio input: {json.dumps(response, indent=4)}')

            transcript = []
            idx = 0

            temp = {
                'speaker': 0,
                'utterance': "",
            }
            transcript.append(temp)

            words = response["results"]["channels"][0]["alternatives"][0]["words"]
            for i in range(len(words)):
                if i == 0:
                    continue

                if words[i - 1]["speaker"] == words[i]["speaker"]:
                    print(transcript)
                    transcript[idx]["utterance"] += words[i]["punctuated_word"] + " "
                else:
                    idx += 1
                    temp = {
                        'speaker': words[i]["speaker"],
                        'utterance': words[i]["punctuated_word"] + " ",
                    }
                    transcript.append(temp)

            return jsonify({
                # 'response': response,
                'transcript': transcript,
            })
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print(f'line {line_number}: {exception_type} - {e}')

        return jsonify({'result': 'Something wrong with Deepgram API'})


@app.route('/llm_initialize', methods=['GET'])
def llm_intialize():
    if request.method == 'GET':
        global tokenizer
        global model
        global device

        # path = "bigscience/bloomz-560m"
        # path = "bigscience/bloomz-1b7"
        path = "bigscience/bloomz-3b"
        # path = "bigscience/bloomz-7b1"

        device = "mps"

        print(f'Loading model started from path: {path}')
        tokenizer = BloomTokenizerFast.from_pretrained(path)
        model = BloomForCausalLM.from_pretrained(path)
        model.to(device)
        print(f'Loading model finished...')

        return jsonify({'result': 'success'})


@app.route('/llm_save', methods=['POST'])
def llm_save():
    if request.method == 'POST':
        print(f'Saving model started...')
        save_path = "/Users/snehalyelmati/Documents/models/"
        model_name = request.form["model_name"]
        tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        model = BloomForCausalLM.from_pretrained(model_name)

        print(f'Save path: {save_path + model_name}')
        model.save_pretrained(save_path + model_name)
        tokenizer.save_pretrained(save_path + model_name)
        print(f'Saving model finished...')

        return jsonify({'result': 'success'})


# TODO: add proper loggers
def llm(input: str):
    # generator = pipeline("text2text-generation", model, tokenizer, device=device)
    # print(f'Pipeline output: {generator(input, max_length=150, num_return_sequences=1)}')

    # inputs = tokenizer('''
    #     Advice from a financial adviser: "Invest in company XYZ, the returns would triple in an year."

    #     Based on this advice, answer the below question,
    #     Question: Is the advice fradulent? If yes, return "yes", else return "no"

    #     Answer: ''', return_tensors="pt")["input_ids"].to(device)

    # inputs = tokenizer('A cat in French is "', return_tensors="pt")["input_ids"].to(device)
    inputs = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(inputs, max_new_tokens=3)
    decoded_output = tokenizer.decode(outputs[0])
    print(f'Decoded output: {decoded_output}')
    return decoded_output


@app.route('/llm_predict', methods=['POST'])
def llm_predict():
    if request.method == 'POST':
        data = request.form["utterance"]
        result = llm(str(data))
        return jsonify({'result': result})


# driver function
if __name__ == '__main__':
    app.run(port=5999, debug=True)

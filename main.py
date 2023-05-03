import asyncio
import json
import os
import sys

from deepgram import Deepgram
from flask import Flask, jsonify, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BloomTokenizerFast, \
    BloomForCausalLM

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

            # Build transcript object using the speech to text response
            idx = 0
            transcript = []
            temp = {'speaker': 0, 'utterance': "", 'fraudulent': False}
            transcript.append(temp)

            words = response["results"]["channels"][0]["alternatives"][0]["words"]
            for i in range(len(words)):
                if i == 0:
                    continue

                if words[i - 1]["speaker"] == words[i]["speaker"]:
                    # print(transcript)
                    transcript[idx]["utterance"] += words[i]["punctuated_word"] + " "
                else:
                    idx += 1
                    temp = {
                        'speaker': words[i]["speaker"],
                        'utterance': words[i]["punctuated_word"] + " ",
                        'fraudulent': False
                    }
                    transcript.append(temp)

            # Pass the response through LLM
            # TODO: use llm()
            for i in range(len(transcript)):
                #             input = f'''This is a segment of conversation between a financial consultant and his client,
                # Conversation: "{transcript[i]['utterance']}"
                # Question: Is the above conversation potentially fraudulent? If it is return "yes" or else return "no".
                # Answer:
                #             '''
                llm_output = llm(transcript[i]['utterance'])

                if llm_output == "yes":
                    transcript[i]['fraudulent'] = True
                    transcript[i]['llm_output'] = llm_output

            return jsonify({
                'response': response,
                'transcript': transcript,
            })
        except Exception as e:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            line_number = exception_traceback.tb_lineno
            print(f'line {line_number}: {exception_type} - {e}')

        return jsonify({'result': 'Something wrong with Deepgram API'})


@app.route('/llm_initialize', methods=['GET'])
def llm_initialize():
    if request.method == 'GET':
        global tokenizer
        global model
        global device

        # path = "bigscience/bloomz-560m"
        # path = "bigscience/bloomz-1b7"
        # path = "bigscience/bloomz-3b"
        # path = "bigscience/bloomz-7b1"

        # path = "declare-lab/flan-alpaca-large"
        # path = "declare-lab/flan-gpt4all-xl"

        # path = "lmsys/vicuna-13b-delta-v1.1"
        path = "lmsys/vicuna-7b-delta-v1.1"

        device = "mps"

        print(f'Loading model started from path: {path}')

        if "bloom" in path:
            tokenizer = BloomTokenizerFast.from_pretrained(path)
            model = BloomForCausalLM.from_pretrained(path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)

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


def llm(input: str, is_full_prompt="False"):
    print("LLM prediction started...")
    # generator = pipeline("text2text-generation", model, tokenizer, device=device)
    # print(f'Pipeline output: {generator(input, max_length=150, num_return_sequences=1)}')

    if is_full_prompt == "True":
        inputs = tokenizer(input, return_tensors="pt")["input_ids"].to(device)
    else:
        advice = input
        # advice = "Invest in company XYZ, the returns would surely triple in a year."
        inputs = tokenizer(
            f'''Assume you are a financial adviser who flags fraudulent advices. Your
task is to review the advice, delimited by <>, given by another
financial advisor to their client.

Question: Is the advice given by the financial adviser fraudulent?

Format your output as a valid JSON object with the following keys,

1. "Reasoning" - reasoning for the question above.
2. "Final answer" - final answer whether the advice is fraudulent. “Yes” if the advice is fraudulent, “No” if it is not fraudulent.

Do not include any additional information apart from the information that is requested above.

Advice: {advice}> 
Output:
'''
            , return_tensors="pt")["input_ids"].to(device)

    # inputs = tokenizer('A cat in French is "', return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(inputs, max_length=1000)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_output)
    print(f'LLM prediction finished!')
    return decoded_output


@app.route('/llm_predict', methods=['POST'])
def llm_predict():
    if request.method == 'POST':
        data = request.form["utterance"]
        is_full_prompt = request.form["is_full_prompt"]
        result = llm(str(data), is_full_prompt)
        return jsonify({'result': result})


# driver function
if __name__ == '__main__':
    app.run(port=5999, debug=True)

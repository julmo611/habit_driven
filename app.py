from flask import Flask, jsonify, request, render_template, url_for
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from dotenv import load_dotenv
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/construct-index', methods=['POST'])
def construct_index():
    data = request.json
    directory_path = data['directory_path']

    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # prompt helper
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit)

    # define LLM predictor
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0, model_name='text-davinci-003', max_tokens=num_outputs))

    # load data from files in the specified directory
    documents = SimpleDirectoryReader(directory_path).load_data()

    # create index
    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # save index to disk
    index.save_to_disk('index.json')

    return jsonify({'message': 'Index constructed successfully'})


@app.route('/ask-bot', methods=['POST'])
def ask_bot():
    data = request.json
    query = data['query']
    input_index = data['input_index']

    index = GPTSimpleVectorIndex.load_from_disk(input_index)
    response = index.query(query, response_mode='compact')

    return jsonify({'response': response.response,  'clearInput': True})


if __name__ == '__main__':
    app.run(debug=True)

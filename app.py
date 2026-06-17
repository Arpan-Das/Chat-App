import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

app = Flask(__name__)

# Load environment variables
load_dotenv()
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT")

# Initialize OpenAI client with Azure identity
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)
openai_client = OpenAI(base_url=azure_openai_endpoint, api_key=token_provider)

last_response_id = None

@app.route("/")
def index():
    return render_template("chat.html")  # simple HTML frontend

@app.route("/chat", methods=["POST"])
def chat():
    global last_response_id
    user_input = request.json.get("message")

    stream = openai_client.responses.create(
        model=model_deployment,
        instructions="You are a helpful AI assistant that answers questions and provides information.",
        input=user_input,
        previous_response_id=last_response_id,
        stream=False
    )

    last_response_id = stream.id
    return jsonify({"response": stream.output_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

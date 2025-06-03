import os
import re
import json
from textwrap import dedent

import requests
from dotenv import load_dotenv
from fake_useragent import UserAgent
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    retry_if_not_exception_type,
)


def load_environment(environment="production"):
    """Load environment API keys as environment variables"""
    if environment == "development":
        env_path = "../.env_dev"
    elif environment == "production":
        env_path = "../.env"
    else:
        raise ValueError(
            f"Invalid environment specified: {environment}. Use 'production' or 'development'."
        )

    load_dotenv(dotenv_path=env_path, override=True)


def run_prompt(
    system_message,
    user_input,
    model,
    environment="production",
    response_format={"type": "json_object"},
):
    load_environment(environment)

    if model.startswith("gpt"):
        response, usage = prompt_gpt(model, system_message, user_input, response_format)
    elif model.startswith("o1"):
        response, usage = prompt_o1(model, system_message, user_input, response_format)
    elif model.startswith("o3"):
        response, usage = prompt_o3(model, system_message, user_input, response_format)
    elif model.startswith(("meta", "llama", "deepseek")):
        if model == "meta-llama/Llama-3.3-70B-Instruct":
            model = "llama3-sdsc"
        if model == "meta-llama/Llama-3.2-90B-Vision-Instruct":
            model = "llama3"
        if model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
            model = "DeepSeek-R1-Distill-Qwen-32B"
        response, usage = prompt_nrp_llml(
            model, system_message, user_input, response_format
        )

    return response, usage


def prompt_gpt(
    model, system_message, user_input, response_format={"type": "json_object"}
):
    client = get_client()
    MAX_TOKENS = 2000

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
        response_format=response_format,
    )
    # Convert token usage to a dictionary
    usage = json.loads(response.usage.json())

    return (response.choices[0].message.content, usage)


def prompt_o1(
    model, system_message, user_input, response_format={"type": "json_object"}
):
    client = get_client()
    MAX_TOKENS = 2000

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": system_message + user_input,
            },  # no system message for o1
        ],
        # temperature=0,
        # {'error': {'message': "Unsupported value: 'temperature' does not support 0 with this model.
        # Only the default (1) value is supported.", 'type': 'invalid_request_error', 'param': 'temperature', 'code': 'unsupported_value'}}
        # response_format={"type": "json_object"},
        # {'error': {'message': "Invalid parameter: 'response_format' of type 'json_schema' is not supported with this model.
        # Learn more about supported models at the Structured Outputs guide: https://platform.openai.com/docs/guides/structured-outputs",
        # 'type': 'invalid_request_error', 'param': None, 'code': None}}
        max_completion_tokens=MAX_TOKENS,
    )
    # Convert token usage to a dictionary
    usage = json.loads(response.usage.json())

    # Format JSON output using gpt-40-mini
    # see: https://cookbook.openai.com/examples/o1/using_chained_calls_for_o1_structured_outputs#structured-outputs
    o1_response_content = response.choices[0].message.content
    gpt_response_content, _ = prompt_gpt(
        "gpt-4o-mini",
        "Format the following text into valid JSON",
        o1_response_content,
        response_format,
    )
    return (gpt_response_content, usage)


def prompt_o3(
    model, system_message, user_input, response_format={"type": "json_object"}
):
    client = get_client()
    MAX_TOKENS = 2000

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ],
        # temperature=0, # Unsupported parameter: 'temperature' is not supported with this model.
        # https://platform.openai.com/docs/guides/reasoning/controlling-costs
        reasoning_effort="low",
        max_completion_tokens=MAX_TOKENS,
        response_format=response_format,
    )
    # Convert token usage to a dictionary
    usage = json.loads(response.usage.json())

    return (response.choices[0].message.content, usage)


# response_format see:
# https://platform.openai.com/docs/guides/structured-outputs?example=structured-data#examples
# https://cookbook.openai.com/examples/structured_outputs_intro
def prompt_sdsc_llml(
    model, system_message, user_input, response_format={"type": "json_object"}
):
    API_KEY = os.environ.get("SDSC_LLM_API_KEY")
    BASE_URL = os.environ.get("SDSC_LLM_URL")
    MAX_TOKENS = 2000
    print("SDSC LLM model", model)
    print("SDSC LLM base_url:", BASE_URL)

    messages = [
        {"role": "system", "content": dedent(system_message)},
        {"role": "user", "content": user_input},
    ]

    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": response_format,
    }

    response = requests.post(
        os.path.join(BASE_URL, "chat/completions"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(data),
        timeout=240,  # setting timeout to avoid run-away chats
    )

    response_json = response.json()

    choice = response_json.get("choices", [{}])[0]
    if choice.get("finish_reason") == "length":
        print(f"Warning: Response was truncated due to max_tokens limit: {MAX_TOKENS}")

    if choice["message"] == "Gateway Time-out":
        print(response_json)

    chat_response = choice["message"]["content"].strip()
    usage = response_json["usage"]

    return chat_response, usage


def prompt_nrp_llml(
    model, system_message, user_input, response_format={"type": "json_object"}
):
    API_KEY = os.environ.get("NRP_LLM_API_KEY")
    BASE_URL = os.environ.get("NRP_LLM_URL")
    MAX_TOKENS = 2000

    messages = [
        {"role": "system", "content": dedent(system_message)},
        {"role": "user", "content": user_input},
    ]

    data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "response_format": response_format,
    }

    response = requests.post(
        os.path.join(BASE_URL, "chat/completions"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(data),
        timeout=240,  # setting timeout to avoid run-away chats
    )

    response_json = response.json()

    choice = response_json.get("choices", [{}])[0]
    if choice.get("finish_reason") == "length":
        print(f"Warning: Response was truncated due to max_tokens limit: {MAX_TOKENS}")

    if choice["message"] == "Gateway Time-out":
        print(response_json)

    if choice["message"].get("content") is not None:
        chat_response = choice["message"]["content"].strip()
    else:
        chat_response = choice["message"].get("reasoning_content", "{}")

    usage = response_json["usage"]

    return chat_response, usage


# https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def get_embedding(text, model="text-embedding-3-small"):
    load_environment()
    client = get_client()
    return client.embeddings.create(input=text, model=model).data[0].embedding


def get_client():
    API_KEY = os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=API_KEY)


def get_open_ai_models():
    client = get_client()
    models = client.models.list()

    # Extract the model ids
    return re.findall(r"Model\(id='([^']*)'", str(models))


def get_sdsc_models():
    API_KEY = os.environ.get("SDSC_LLM_API_KEY")
    BASE_URL = os.environ.get("SDSC_LLM_URL")

    response = requests.get(
        os.path.join(BASE_URL, "models"), headers={"Authorization": f"Bearer {API_KEY}"}
    )
    json_response = response.json()

    # Extract the model ids
    return [item.get("id") for item in json_response.get("data", [])]


def get_nrp_models():
    API_KEY = os.environ.get("NRP_LLM_API_KEY")
    BASE_URL = os.environ.get("NRP_LLM_URL")

    response = requests.get(
        os.path.join(BASE_URL, "models"), headers={"Authorization": f"Bearer {API_KEY}"}
    )
    json_response = response.json()

    # Extract the model ids
    return [item.get("id") for item in json_response.get("data", [])]


def get_models(environment="production"):
    load_environment(environment)
    open_ai_models = get_open_ai_models()
    sdsc_models = get_sdsc_models()
    nrp_models = get_nrp_models()

    return {
        "OPEN_AI": open_ai_models,
        "SDSC_LLM": sdsc_models,
        "NRP_LLM": nrp_models,
    }


def get_token_count(text, model):
    if model.startswith("meta") or model.startswith("mistralai"):
        return 0

    # Load the encoding
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text
    tokens = encoding.encode(text)

    # Calculate the number of tokens
    return len(tokens)


def get_token_cost(num_tokens, model, direction):
    # Pricing see: https://openai.com/api/pricing/
    # **Output tokens include internal reasoning tokens generated by the model that are not visible in API responses.
    if model == "gpt-4o":
        if direction == "input":
            return 2.5 * num_tokens / 1000000  # $2.5 per 1,000,000 tokens
        if direction == "output":
            return 10.0 * num_tokens / 1000000  # $10.0 per 1,000,000 tokens
    if model == "gpt-4o-mini":
        if direction == "input":
            return 0.15 * num_tokens / 1000000  # $0.15 per 1,000,000 tokens
        if direction == "output":
            return 0.60 * num_tokens / 1000000  # $0.60 per 1,000,000 tokens
    if model == "o1-preview":
        if direction == "input":
            return 15.0 * num_tokens / 1000000  # $15.0 per 1,000,000 tokens
        if direction == "output":
            return 60.0 * num_tokens / 1000000  # $60.0 per 1,000,000 tokens
    if model == "o1-mini":
        if direction == "input":
            return 3.0 * num_tokens / 1000000  # $3.0 per 1,000,000 tokens
        if direction == "output":
            return 12.0 * num_tokens / 1000000  # $12.0 per 1,000,000 tokens **
    if model == "o3-mini-2025-01-31":
        if direction == "input":
            return 1.1 * num_tokens / 1000000  # $1.10 per 1,000,000 tokens
        if direction == "output":
            return 4.4 * num_tokens / 1000000  # $4.40 per 1,000,000 tokens **
    if model.startswith("meta") or model.startswith("deepseek"):
        return 0.0  # "free" runs at SDSC

    return 0.0


def load_pdf(pdf_path, keep_references=False):
    loader = PyPDFLoader(file_path=pdf_path)
    pages = loader.load()
    text = "\n".join(page.page_content for page in pages)

    if not keep_references:
        text = remove_references(text)

    return text


def remove_references(full_text):
    references_start = None

    # Look for "References" or "Reference" at the start of a line
    references_start = None
    for match in re.finditer(r"\n(?:References|Reference)\n", full_text, re.IGNORECASE):
        # Consider this as the start of the References section
        references_start = match.start() + 1  # +1 to skip the preceding newline
        break  # Take the first match found

    # Remove the "References" section
    if references_start != -1:
        cleaned_text = full_text[:references_start]
    else:
        cleaned_text = full_text

    return cleaned_text

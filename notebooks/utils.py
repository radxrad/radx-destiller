import os
import re
import tempfile
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import tiktoken
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Combine the text from the pages
        text = "\n".join(page.page_content for page in pages)

        return text
    except IOError as io_err:
        print(f"IO error: {io_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return "The article is not available! Try to run this method behind your institituional firewall."


def load_pdf_from_pmc(pmcid, timeout=30):
    try:
        # Access PubMed Central webpage using the PMCID
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

        # Generate a random user agent
        ua = UserAgent()
        headers = {"User-Agent": ua.random}
        response = requests.get(pmc_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Parse the PubMed Central webpage to find the PDF link
        soup = BeautifulSoup(response.content, "html.parser")
        pdf_link = None

        # Find the specific section with the PDF link
        pdf_section = soup.find("div", class_="scroller")
        if pdf_section:
            pdf_tag = pdf_section.find("li", class_="pdf-link").find("a", href=True)
            if pdf_tag:
                pdf_link = pdf_tag["href"]

        if not pdf_link:
            return "PDF link not found."

        # Complete the URL
        pdf_url = "https://www.ncbi.nlm.nih.gov/" + pdf_link

        # Download the PDF content
        response = requests.get(pdf_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Save the PDF content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(response.content)
            temp_pdf_path = temp_pdf.name

        # Load the PDF file
        loader = PyPDFLoader(file_path=temp_pdf_path)
        pages = loader.load()

        # Combine the text from the pages
        text = "\n".join(page.page_content for page in pages)

        return text
    except requests.exceptions.RequestException as req_err:
        print(f"Request error: {req_err}")
    except IOError as io_err:
        print(f"IO error: {io_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return "The article is not available! Try to run this method behind your institituional firewall."


def get_available_models(client):
    models = client.models.list()
    # Extract the model ids
    model_ids = re.findall(r"Model\(id='([^']*)'", str(models))

    return model_ids


def get_token_count(text, model):
    # for llama model, use the "gpt-4o-mini" as a proxy
    if model.startswith("meta-llama"):
        model = "gpt-4o-mini"

    # Load the encoding
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text
    tokens = encoding.encode(text)

    # Calculate the number of tokens
    return len(tokens)


def get_token_cost(num_tokens, model):
    # Pricing see: https://platform.openai.com/docs/models
    if model == "gpt-4o":
        return 5.0 * num_tokens / 1000000  # $5.0 per 1,000,000 tokens
    if model == "gpt-4o-mini":
        return 0.15 * num_tokens / 1000000  # $0.15 per 1,000,000 tokens
    if model.startswith("meta-llama"):
        return 0.0  # "free" runs at SDSC

    return 0.0


def prompt_gpt(client, model, user_input, system_message, assistant_message):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
        timeout=120,  # doesn't seem to have any effect
    )

    return response.choices[0].message.content + "\n"


def download_pdf_from_pmc(pmcid, output_folder, timeout=30):
    # Access PubMed Central webpage using the PMCID
    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    print(pmc_url)

    # Generate a random user agent
    ua = UserAgent()
    headers = {"User-Agent": ua.random}
    response = requests.get(pmc_url, headers=headers, timeout=timeout)
    response.raise_for_status()

    # Parse the PubMed Central webpage to find the PDF link
    soup = BeautifulSoup(response.content, "html.parser")
    pdf_link = None

    # Find the specific section with the PDF link
    pdf_section = soup.find("div", class_="scroller")
    if pdf_section:
        pdf_tag = pdf_section.find("li", class_="pdf-link").find("a", href=True)
        if pdf_tag:
            pdf_link = pdf_tag["href"]

    if not pdf_link:
        print("PDF link not found.")
        return

    # Complete the URL
    pdf_link = "https://www.ncbi.nlm.nih.gov/" + pdf_link

    print("pdf_url:", pdf_link)

    # Step 3: Download the PDF
    pdf_response = requests.get(pdf_link, headers=headers, timeout=timeout)
    response.raise_for_status()

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, f"{pmcid}.pdf")
    with open(pdf_path, "wb") as file:
        file.write(pdf_response.content)
        print(f"PDF downloaded successfully: {pdf_path}")

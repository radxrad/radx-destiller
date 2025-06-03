import time

import requests
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import pandas as pd


def get_project_serial_num(project_num):
    project_serial_num = project_num.rsplit("-")[0]
    return project_serial_num[-8:]


def get_award_type(project_num):
    return project_num[1:4]


def is_supplement(project_num):
    return project_num[-2:-1] == "S"


def get_dbgap_dataframe(search_term):
    """
    Query the dbGaP FHIR API for studies whose title contains `search_term`,
    then return a pandas DataFrame with columns ['accession', 'title', 'description'].
    """
    base_url = "http://dbgap-api.ncbi.nlm.nih.gov/fhir/x1/ResearchStudy"
    params = {"title:contains": search_term}

    studies_list = []
    next_url = base_url

    while next_url:
        if next_url == base_url:
            resp = requests.get(next_url, params=params)
        else:
            resp = requests.get(next_url)
        resp.raise_for_status()
        bundle = resp.json()

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            accession_id = resource.get("id")  # e.g., "phs002964.v1.p1"
            title = resource.get("title", "")  # e.g., "RADx-rad COVID-19 Study"
            description = resource.get(
                "description", ""
            )  # new field: study description
            if accession_id:
                studies_list.append(
                    {
                        "accession": accession_id,
                        "title": title,
                        "description": description.strip(),
                    }
                )

        next_link = None
        for link_obj in bundle.get("link", []):
            if link_obj.get("relation") == "next":
                next_link = link_obj.get("url")
                break

        next_url = next_link

    df = pd.DataFrame(studies_list, columns=["accession", "title", "description"])
    return df


def get_dbgap_accessions(search_term):
    """
    Retrieve all dbGaP accession numbers matching `search_term` (e.g., "radx-rad")
    by querying the dbGaP FHIR API.
    See https://github.com/ncbi/DbGaP-FHIR-API-Docs
    """
    base_url = "http://dbgap-api.ncbi.nlm.nih.gov/fhir/x1/ResearchStudy"
    # Use title:contains for substring matching (FHIR string search)
    # URL‐encode spaces and special characters automatically via params
    params = {
        "title:contains": search_term  # searches for studies whose title includes 'radx-rad'
        # "keyword": search_term
    }

    accessions = set()
    next_url = base_url  # start with base endpoint

    while next_url:
        # If next_url already has query params, pass params=None
        if next_url == base_url:
            response = requests.get(next_url, params=params)
        else:
            response = requests.get(next_url)
        response.raise_for_status()

        bundle = response.json()

        # Each entry in bundle["entry"] holds a Resource with "id"
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            accession_id = resource.get("id")
            if accession_id:
                accessions.add(accession_id)

        # Determine the "next" link (for pagination)
        next_link = None
        for link in bundle.get("link", []):
            if link.get("relation") == "next":
                next_link = link.get("url")
                break

        next_url = next_link  # if None, loop will exit

    return sorted(accessions)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def extract_dbgap_study_info(dbgap_accession):
    time.sleep(1)
    # Compose URL for dbgap API
    # phs003124.v1.p1 -> phs003124
    study_id = dbgap_accession.split(".")[0]
    url = f"https://dbgap-api.ncbi.nlm.nih.gov/fhir/x1/ResearchStudy/?_format=json&_id={study_id}"

    # Fetch the JSON data from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the request failed
    data = response.json()

    # Navigate to the resource object that contains the desired fields
    resource = data["entry"][0]["resource"]

    # Extract the identifier
    # identifier = resource["identifier"][0]["value"] if "identifier" in resource and len(resource["identifier"]) > 0 else None

    # Extract the title
    title = resource.get("title", "")
    title = title.replace("\n", "")

    # Extract the focus text
    study_focus = (
        resource["focus"][0]["text"]
        if "focus" in resource and len(resource["focus"]) > 0
        else ""
    )

    # Extract the description
    description = resource.get("description", "")
    description = description.replace("\n", "").replace("<br>", "")
    print(dbgap_accession)

    return dbgap_accession, title, study_focus, description

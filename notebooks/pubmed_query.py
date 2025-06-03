# Filename: pubmed_query.py
# Author: Christian Horgan
# Created: 2023-12-05
# Description: Retrieve metadata for publications from PubMed.

import os
import time
from typing import Dict, Any

import requests
from fake_useragent import UserAgent
import pandas as pd
from tenacity import (
    retry,
    wait_exponential,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_not_exception_type,
)

from Bio import Entrez


CACHE_PATH = "../.cache"

# Replace this with your email address
Entrez.email = "your.email@example.com"  # NCBI requires this


def safe_get(data: Dict, keys: list, default: Any = "") -> Any:
    """Safely navigate nested dictionaries."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def fetch_grant_articles(grant: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch PubMed articles for a specific grant with error handling."""
    try:
        with Entrez.esearch(
            db="pubmed",
            term=f"{grant}[Grant Number]",
            mindate=start_year,
            maxdate=end_year,
            retmax=1000,
            usehistory="y",
        ) as search:
            result = Entrez.read(search)

        if not result["IdList"]:
            return pd.DataFrame()

        with Entrez.efetch(
            db="pubmed",
            retmode="xml",
            webenv=result["WebEnv"],
            query_key=result["QueryKey"],
        ) as fetch:
            articles = Entrez.read(fetch)

        data = []
        for article in articles["PubmedArticle"]:
            pmid = safe_get(article, ["MedlineCitation", "PMID"])
            med_cit = safe_get(article, ["MedlineCitation"], {})
            art_meta = safe_get(med_cit, ["Article"], {})

            # Extract article IDs from PubmedData
            article_ids = {
                id_item.attributes.get("IdType", "unknown"): str(id_item)
                for id_item in safe_get(article, ["PubmedData", "ArticleIdList"], [])
            }

            # Extract publication date info
            pub_date = safe_get(art_meta, ["Journal", "JournalIssue", "PubDate"], {})
            publication_date = format_publication_date(pub_date)
            year = pub_date.get("Year") or (
                pub_date.get("MedlineDate", "")[:4]
                if pub_date.get("MedlineDate")
                else ""
            )

            kw_list = safe_get(med_cit, ["KeywordList"], [])
            keywords = [str(kw) for kw in kw_list[0]] if kw_list else []

            # Extract MeSH terms and mesh IDs
            mesh_headings = safe_get(med_cit, ["MeshHeadingList"], [])
            mesh_terms = [
                str(mesh["DescriptorName"])
                for mesh in mesh_headings
                if "DescriptorName" in mesh
            ]
            mesh_ids = [
                mesh["DescriptorName"].attributes.get("UI", "")
                for mesh in mesh_headings
                if "DescriptorName" in mesh
            ]

            authors = [
                f"{a.get('LastName', '')}, {a.get('ForeName', '')}".strip(", ")
                for a in safe_get(art_meta, ["AuthorList"], [])
            ]

            data.append(
                {
                    "pm_id": pmid,
                    "pmc_id": article_ids.get("pmc", ""),
                    "doi": article_ids.get("doi", ""),
                    "title": safe_get(art_meta, ["ArticleTitle"]),
                    "abstract": " ".join(
                        safe_get(art_meta, ["Abstract", "AbstractText"], [])
                    ),
                    "keywords": keywords,
                    "mesh_ids": mesh_ids,
                    "mesh_terms": mesh_terms,
                    "authors": authors,
                    "journal": safe_get(art_meta, ["Journal", "Title"]),
                    "year": year,
                    "publication_date": publication_date,
                    "article_type": safe_get(art_meta, ["PublicationTypeList"], []),
                    "grant_number": grant,  # Passed-in grant number added here
                }
            )

        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error processing grant {grant}: {e}")
        return pd.DataFrame()


def get_pubmed_data_new(
    grant_list: list, start_year: int, end_year: int
) -> pd.DataFrame:
    """Retrieve PubMed data for multiple grants."""
    frames = []
    for grant in grant_list:
        # print(f"Processing grant {grant}...", end=)
        df = fetch_grant_articles(grant, start_year, end_year)
        if not df.empty:
            frames.append(df)
        print(f"Processed grant {grant}: {df.shape[0]} publications")
        time.sleep(0.34)  # Respect NCBI rate limit of ~3 requests/sec
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def format_publication_date(pub_date: Dict) -> str:
    """
    Convert publication date info to YYYY-MM-DD format.
    If Year exists, use Month and Day if available, otherwise default to "01".
    If only MedlineDate is provided, attempt to extract the year.
    """
    if pub_date.get("Year"):
        year = pub_date.get("Year", "")
        # Default month and day to "01" if not provided
        month_str = pub_date.get("Month", "01")
        day_str = pub_date.get("Day", "01")

        # Map common month abbreviations to numbers
        month_mapping = {
            "Jan": "01",
            "Feb": "02",
            "Mar": "03",
            "Apr": "04",
            "May": "05",
            "Jun": "06",
            "Jul": "07",
            "Aug": "08",
            "Sep": "09",
            "Oct": "10",
            "Nov": "11",
            "Dec": "12",
        }

        # Convert month to two-digit numeric format
        if month_str.isdigit():
            month = month_str.zfill(2)
        else:
            month = month_mapping.get(month_str[:3], "01")

        # Ensure day is two digits
        day = day_str.zfill(2) if day_str.isdigit() else "01"

        return f"{year}-{month}-{day}"
    else:
        # Fallback: try to extract year from MedlineDate if available
        medline_date = pub_date.get("MedlineDate", "")
        if medline_date and medline_date[:4].isdigit():
            return f"{medline_date[:4]}-01-01"
        return ""


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def download_pmc_pdf(pmc_id):
    # The @retry decorator from tenacity is used to wrap the download_pdf function.
    # stop_after_attempt(5) specifies that the function should stop retrying after 5 attempts.
    # wait_exponential(multiplier=1, min=2, max=60) sets the exponential backoff strategy
    # with a minimum wait time of 2 seconds and a maximum wait time of 60 seconds.
    # retry_if_exception_type(requests.exceptions.RequestException) ensures that the function
    # retries only on exceptions related to the requests library.

    pdf_filename = f"{pmc_id}.pdf"
    pdf_path = os.path.join(CACHE_PATH, pdf_filename)

    # Skip files already downloaded
    if os.path.exists(pdf_path):
        return

    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"

    # Generate a random user agent. PMC will not accept requests without a user agent.
    ua = UserAgent()
    headers = {"User-Agent": ua.random}

    response = requests.get(pdf_url, headers=headers, timeout=60)
    response.raise_for_status()

    with open(pdf_path, "wb") as file:
        file.write(response.content)
    print(f"Downloaded: {pmc_id}")
    time.sleep(5)  # Base delay between requests

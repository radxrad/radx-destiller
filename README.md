# radx-destiller

**radx-destiller** identifies and ranks publications relevant to the [NIH RADx-rad](https://www.nih.gov/research-training/medical-research-initiatives/radx/radx-programs#radx-rad) initiative using Large Language Models (LLMs).

The tool evaluates publications citing RADx-rad grant numbers to assess alignment with:

* **Funding Opportunity Announcements (FOAs)** – using full-text descriptions
* **dbGaP Study Objectives** – based on titles and abstracts

LLMs compare each publication’s title and abstract to FOA and dbGaP content, assigning a 5-point Likert relevance score along with a rationale for the rating. Publications are excluded from the evaluation if the length of the publication title + abstract is less than 100 characters.

This study used the `meta-llama/Llama-3.3-70B-Instruct` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` LLMs hosted at UCSD/San Diego Supercomputer Center, however, the code also supports the OpenAI models.

### Output

The tool generates the following files, each containing the Likert score and rationale from the LLM evaluation:

| Filename                                   | Description                                                           |
| ------------------------------------------ | --------------------------------------------------------------------- |
| `publication/radx_rad_related_publications_YYYY-MM-DD` | Publications that reference RADx-rad grants that are directly aligned with RADx-rad program objectives        |
| `publications/radx_rad_other_publications_YYYY-MM-DD`   | Publications that reference RADx-rad grants but lack direct relevance |
| `results/DeepSeek-R1-Distill-Qwen-32B/annotation_full_text_5` | Directory of files with raw LLM evaluations |
| `results/Llama-3.3-70B-Instruct/annotation_full_text_5` | Directory of files with raw LLM evaluations |


---

### Setup

------
Prerequisites: Miniconda3 (light-weight, preferred) or Anaconda3 and Mamba (faster than Conda)

* Install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)
* Update an existing miniconda3 installation: ```conda update conda```
* Install Mamba: ```conda install mamba -n base -c conda-forge```
* Install Git (if not installed): ```conda install git -n base -c anaconda```
------

1. Clone this Repository

```
git clone https://github.com/radxrad/radx-destiller
cd radx-destiller
```

2. Create a Conda environment

The file `environment.yml` specifies the Python version and all required dependencies.

```
mamba env create -f environment.yml
```

3. Setup OpenAI and/or SDSC LLM API Keys
```
cp env_template .env
```
Then edit the .env file and set your API keys


### Run the Notebooks

------
1. Activate the conda environment

```
conda activate radx-destiller
```

2. Start `jupyter lab` and navigate to the `notebooks` directory

```
jupyter lab
```

3. Run the notebooks in the listed order to create the lists of RADx-rad publications.
   
3. When you are finished, deactivate the conda environment

```
conda deactivate
```

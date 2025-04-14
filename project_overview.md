### Directory Structure
```bash
/home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink
├── eval
│   ├── Information_Diversity.py
│   ├── Knowledge_Density.py
│   ├── Rubric_Grading_new.py
│   ├── Rubric_Grading.py
│   └── trim.py
├── examples
│   ├── deepseekr1.py
│   ├── gpt4o_agent.py
│   └── gpt4o.py
└── src
    ├── actions
    │   ├── article_generation.py
    │   ├── article_polish.py
    │   ├── __init__.py
    │   └── outline_generation.py
    ├── dataclass
    │   ├── Article.py
    │   ├── __init__.py
    │   └── interface.py
    ├── __init__.py
    ├── tools
    │   ├── __init__.py
    │   ├── lm.py
    │   ├── mindmap.py
    │   └── rm.py
    └── utils
        ├── ArticleTextProcessing.py
        ├── FileIOHelper.py
        ├── __init__.py
        ├── post.py
        ├── utils.py
        └── WebPageHelper.py

7 directories, 26 files

```

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/eval/Information_Diversity.py
- `import numpy as np`
- `import os`
- `import json`
- `from sentence_transformers import SentenceTransformer`
- `from sklearn.metrics.pairwise import cosine_similarity`
- `from argparse import ArgumentParser`

- `def get_snippets()`
- `def calculate_snippet_similarities()`
- `def main()`
- `def calculate_deepthink()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/eval/trim.py
- `import re`

- **Class: `ArticleNode`**
  - `__init__()`
  - `def add_content()`
  - `def add_child()`
  - `def __repr__()`

- `def parse_article()`
- `def iterative_trim()`
- `def process_document()`
- `def reconstruct_article()`
- `def update_count()`
- `def text_word_count()`
- `def find_node_with_lowest_length()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/eval/Rubric_Grading.py
- `import os`
- `from argparse import ArgumentParser`
- `from concurrent.futures import ThreadPoolExecutor, as_completed`
- `from prometheus_eval import PrometheusEval`
- `from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE`
- `from prometheus_eval.vllm import VLLM`
- `import os`
- `import logging`
- `import argparse`
- `import pandas as pd`
- `from tqdm import tqdm`
- `from collections import defaultdict`
- `from eval.bin.utils import process_document`
- `from eval.metrics import article_entity_recall, compute_rouge_scores`
- `from config.paths import hf_cache_dir`
- `from src.utils import dump_json as save_results`
- `from src.utils import (`
- `import sys`

- **Class: `ComprehensiveEvaluator`**
  - `__init__()`
  - `def evaluate_files()`
  - `def grade()`
  - `def Coverage()`
  - `def process_file()`
  - `def Novelty()`
  - `def Depth()`
  - `def Relevance()`

- `def main()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/eval/Rubric_Grading_new.py
- `import os`
- `import json`
- `import re`
- `import numpy as np`
- `from argparse import ArgumentParser`
- `from concurrent.futures import ThreadPoolExecutor, as_completed`
- `from prometheus_eval import PrometheusEval`
- `from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE`
- `from FActScore.factscore.atomic_facts import AtomicFactGenerator, normalize_answer`
- `from prometheus_eval.litellm import LiteLLM`
- `from prometheus_eval.vllm import VLLM`
- `from trim import process_document`

- **Class: `ComprehensiveEvaluator`**
  - `__init__()`
  - `def evaluate_files()`
  - `def grade()`
  - `def Coverage()`
  - `def process_file()`
  - `def Novelty()`
  - `def Depth()`
  - `def Relevance()`

- `def main()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/eval/Knowledge_Density.py
- `import os`
- `import numpy as np`
- `from argparse import ArgumentParser`
- `from FActScore.factscore.atomic_facts import AtomicFactGenerator, normalize_answer`
- `from trim import process_document`
- `from ..src.tools.lm import *`

- `def deduplicate_atomic_facts()`
- `def main()`
- `def knowledge_density_grade()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/examples/deepseekr1.py
- `import os`
- `from argparse import ArgumentParser`
- `from src.tools.lm import DeepSeekModel`
- `from src.tools.rm import GoogleSearchAli`
- `from src.tools.mindmap import MindMap`
- `from src.actions.outline_generation import OutlineGenerationModule`
- `from src.dataclass.Article import Article`
- `from src.actions.article_generation import ArticleGenerationModule`
- `from src.actions.article_polish import ArticlePolishingModule`

- `def main()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/examples/gpt4o.py
- `import os`
- `import sys`
- `from argparse import ArgumentParser`
- `from src.tools.lm import OpenAIModel_dashscope`
- `from src.tools.rm import GoogleSearchAli`
- `from src.tools.mindmap import MindMap`
- `from src.actions.outline_generation import OutlineGenerationModule`
- `from src.dataclass.Article import Article`
- `from src.actions.article_generation import ArticleGenerationModule`
- `from src.actions.article_polish import ArticlePolishingModule`

- `def main()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/examples/gpt4o_agent.py
- `import os`
- `import sys`
- `import yaml`
- `import re`
- `from argparse import ArgumentParser`
- `from ..src.actions.article_generation import ArticleGenerationModule`
- `from ..src.actions.article_polish import ArticlePolishingModule`
- `from ..src.actions.outline_generation import OutlineGenerationModule`
- `from ..src.dataclass.Article import Article`
- `from ..src.tools.lm import OpenAIModel_dashscope`
- `from ..src.tools.mindmap import MindMap`
- `from ..src.tools.rm import GoogleSearchAli`

- `def load_config()`
- `def extract_agent_arguments()`
- `def main()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/__init__.py
- `from .utils import *`
- `from .dataclass import *`
- `from .actions import *`
- `from .tools import *`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/actions/article_polish.py
- `import copy`
- `import dspy`
- `from src.utils.ArticleTextProcessing import ArticleTextProcessing`

- **Class: `ArticlePolishingModule`**
  - `__init__()`
  - `def polish_article()`
- **Class: `PolishPage`**
  - `__init__()`
- **Class: `PolishPageModule`**
  - `__init__()`
  - `def forward()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/actions/outline_generation.py
- `import dspy`
- `from src.tools.mindmap import MindMap`
- `from src.utils.ArticleTextProcessing import ArticleTextProcessing`

- **Class: `OutlineGenerationModule`**
  - `__init__()`
  - `def generate_outline()`
- **Class: `WriteOutline`**
  - `__init__()`
  - `def forward()`
- **Class: `PolishPageOutline`**
  - `__init__()`
- **Class: `WritePageOutline`**
  - `__init__()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/actions/__init__.py
- `from .article_generation import *`
- `from .article_polish import *`
- `from .outline_generation import *`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/actions/article_generation.py
- `import concurrent.futures`
- `import copy`
- `import logging`
- `import random`
- `from concurrent.futures import as_completed`
- `from typing import List, Union`
- `import random`
- `import dspy`
- `import sys`
- `from src.utils.ArticleTextProcessing import ArticleTextProcessing`

- **Class: `ArticleGenerationModule`**
  - `__init__()`
  - `def generate_section()`
  - `def generate_article()`
- **Class: `ConvToSection`**
  - `__init__()`
  - `def forward()`
- **Class: `WriteSection`**
  - `__init__()`
- **Class: `WriteSectionAgentEnglish`**
  - `__init__()`
- **Class: `WriteSectionAgentChinese`**
  - `__init__()`
- **Class: `WriteSectionAgentFormalChinese`**
  - `__init__()`
- **Class: `WriteSectionAgentEnthusiasticChinese`**
  - `__init__()`
- **Class: `WriteSectionAgentEnthusiasticEnglish`**
  - `__init__()`
- **Class: `WriteSectionAgentFormalEnglish`**
  - `__init__()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/dataclass/Article.py
- `import copy`
- `import re`
- `from typing import Optional, List, Dict`
- `from src.dataclass.interface import articleSectionNode, article`
- `from src.utils.ArticleTextProcessing import ArticleTextProcessing`
- `from src.utils.FileIOHelper import FileIOHelper`

- **Class: `Article`**
  - `__init__()`
  - `def get_outline()`
  - `def get_outline_tree()`
  - `def dump_article_as_plain_text()`
  - `def post_processing()`
  - `def insert_or_create_section()`
  - `def update_section()`
  - `def to_string()`
  - `def find_section()`
  - `def get_leaf_nodes()`
  - `def get_first_level_section_names()`
  - `def _merge_new_info_to_references()`
  - `def from_outline_str()`
  - `def reorder_reference_index()`
  - `def dump_reference_to_file()`
  - `def get_outline_as_list()`
  - `def dump_outline_to_file()`

- `def pre_order_update_index()`
- `def traverse()`
- `def build_tree()`
- `def pre_order_find_index()`
- `def _get_all_section_names()`
- `def preorder_traverse()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/dataclass/__init__.py
- `from .Article import *`
- `from .interface import *`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/dataclass/interface.py
- `import functools`
- `import logging`
- `import time`
- `from abc import ABC, abstractmethod`
- `from collections import OrderedDict`
- `from typing import Dict, List, Optional, Union`

- **Class: `Information`**
  - `__init__()`
- **Class: `InformationTable`**
  - `__init__()`
  - `def retrieve_information()`
- **Class: `articleSectionNode`**
  - `__init__()`
  - `def remove_child()`
  - `def add_child()`
- **Class: `article`**
  - `__init__()`
  - `def get_outline_tree()`
  - `def to_string()`
  - `def find_section()`
  - `def get_first_level_section_names()`
  - `def prune_empty_nodes()`
- **Class: `Retriever`**
  - `__init__()`
  - `def retrieve()`
  - `def update_search_top_k()`
  - `def collect_and_reset_rm_usage()`
- **Class: `KnowledgeCurationModule`**
  - `__init__()`
  - `def research()`
- **Class: `OutlineGenerationModule`**
  - `__init__()`
  - `def generate_outline()`
- **Class: `articleGenerationModule`**
  - `__init__()`
  - `def generate_article()`
- **Class: `articlePolishingModule`**
  - `__init__()`
  - `def polish_article()`
- **Class: `LMConfigs`**
  - `__init__()`
  - `def collect_and_reset_lm_history()`
  - `def log()`
  - `def collect_and_reset_lm_usage()`
  - `def init_check()`
- **Class: `Engine`**
  - `__init__()`
  - `def run_article_generation_module()`
  - `def run_article_polishing_module()`
  - `def apply_decorators()`
  - `def summary()`
  - `def run()`
  - `def log_execution_time_and_lm_rm_usage()`
  - `def reset()`
  - `def run_knowledge_curation_module()`
  - `def run_outline_generation_module()`

- `def log_execution_time()`
- `def wrapper()`
- `def build_tree()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/tools/lm.py
- `import random`
- `import threading`
- `import time`
- `import dspy`
- `import os`
- `from openai import OpenAI`
- `from zhipuai import ZhipuAI`
- `from typing import Optional, Literal, Any`
- `from dashscope import Generation`
- `import requests`

- **Class: `OpenAIModel_dashscope`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def log_usage()`
  - `def __call__()`
- **Class: `DeepSeekModel`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def log_usage()`
  - `def __call__()`
- **Class: `QwenModel`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def log_usage()`
  - `def __call__()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/tools/mindmap.py
- `import concurrent.futures`
- `import os`
- `import re`
- `import json`
- `import dspy`
- `import numpy as np`
- `import networkx as nx`
- `import matplotlib.pyplot as plt`
- `from typing import Union, List, Optional, Dict`
- `from sentence_transformers import SentenceTransformer`
- `from sklearn.metrics.pairwise import cosine_similarity`
- `from src.utils.ArticleTextProcessing import ArticleTextProcessing`
- `from pipeline.apollo.src.tools import Retriever`
- `from pipeline.apollo.src.tools import LLM`
- `import dspy`
- `import argparse`
- `from pipeline.apollo.src import LLM`
- `from pipeline.apollo.src import VectorRM, Retriever`

- **Class: `ConceptGenerator`**
  - `__init__()`
  - `def forward()`
- **Class: `ExtendConcept`**
  - `__init__()`
- **Class: `GenConcept`**
  - `__init__()`
- **Class: `MindPoint`**
  - `__init__()`
  - `def extend()`
- **Class: `MindMap`**
  - `__init__()`
  - `def prepare_table_for_retrieval()`
  - `def export_categories_and_concepts()`
  - `def visualize_map()`
  - `def get_all_infos()`
  - `def retrieve_information()`
  - `def build_map()`
  - `def load_map()`
  - `def save_map()`
  - `def recursive_extend()`

- `def traverse()`
- `def main()`
- `def _setup_retrieval()`
- `def serialize_node()`
- `def deserialize_node()`
- `def add_edges()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/tools/rm.py
- `import logging`
- `import os`
- `from typing import Callable, Union, List`
- `import dspy`
- `import requests`
- `import re`
- `import uuid`
- `import json`
- `import random`
- `import time`
- `from src.utils.WebPageHelper import WebPageHelper`

- **Class: `GoogleSearchAli`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def forward()`
- **Class: `BingSearchAli`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def forward()`
- **Class: `BingSearch`**
  - `__init__()`
  - `def get_usage_and_reset()`
  - `def forward()`

- `def clean_text()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/tools/__init__.py
- `from .lm import *`
- `from .mindmap import *`
- `from .rm import *`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/ArticleTextProcessing.py
- `import re`
- `from typing import List, Dict`

- **Class: `ArticleTextProcessing`**
  - `__init__()`
  - `def get_first_section_dict_and_list()`
  - `def parse_article_into_dict()`
  - `def limit_word_count_preserve_newline()`
  - `def update_citation_index()`
  - `def clean_up_citation()`
  - `def remove_citations()`
  - `def clean_up_outline()`
  - `def parse_citation_indices()`
  - `def remove_uncompleted_sentences_with_citations()`
  - `def clean_up_section()`

- `def replace_with_individual_brackets()`
- `def deduplicate_group()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/FileIOHelper.py
- `import json`
- `import pickle`

- **Class: `FileIOHelper`**
  - `__init__()`
  - `def dump_pickle()`
  - `def load_pickle()`
  - `def load_str()`
  - `def write_str()`
  - `def dump_json()`
  - `def load_json()`
  - `def handle_non_serializable()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/WebPageHelper.py
- `import concurrent.futures`
- `from typing import List, Dict`
- `import httpx`
- `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- `from trafilatura import extract`

- **Class: `WebPageHelper`**
  - `__init__()`
  - `def urls_to_articles()`
  - `def urls_to_snippets()`
  - `def download_webpage()`


### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/post.py
- `import re`
- `import json`

- `def extract_citations()`
- `def load_map()`
- `def add_ref()`
- `def remove_lines_after_marker()`
- `def remove_consecutive_duplicate_citations()`
- `def polish()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/utils.py
- `import json`
- `import pickle`

- **Class: `FileIOHelper`**
  - `__init__()`
  - `def dump_pickle()`
  - `def load_pickle()`
  - `def load_str()`
  - `def write_str()`
  - `def dump_json()`
  - `def load_json()`
  - `def handle_non_serializable()`

- `def makeStringRed()`

### /home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/src/utils/__init__.py
- `from .ArticleTextProcessing import *`


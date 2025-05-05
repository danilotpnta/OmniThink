# engine.py
import os
import dspy
import json
from dataclasses import dataclass, field
from typing import Optional
from .dataclass.interface import LMConfigs, Engine

from pipeline.omnithink.src.tools.mindmap import MindMap
from pipeline.omnithink.src.dataclass.Article import Article
from pipeline.omnithink.src.utils.FileIOHelper import FileIOHelper
from pipeline.omnithink.src.actions.outline_generation import OutlineGenerationModule
from pipeline.omnithink.src.actions.article_generation import ArticleGenerationModule
from pipeline.omnithink.src.actions.article_polish import ArticlePolishingModule

from pipeline.apollo.src import Retriever


class OmniThinkLLMConfigs(LMConfigs):

    def __init__(self):
        self.lm = None

    def set_lm(self, model: dspy.LM):
        self.lm = model


@dataclass
class OmniThinkRunnerArguments:

    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    search_top_k: int = field(
        default=5,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=6,
        metadata={
            "help": "Maximum number of threads to use. "
            "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."
        },
    )
    embedding_model: str = field(
        default="paraphrase-MiniLM-L6-v2",
        metadata={
            "help": "Embedding model used for the StormInformationTable to store the information collected during KnowledgeCuration stage."
        },
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Random seed for deterministic execution"},
    )
    depth: int = field(
        default=3,
        metadata={
            "help": "The depth of the mind map. "
            "The higher the depth, the more detailed the mind map will be."
        },
    )


class OmniThinkRunner(Engine):
    """STORM Wiki pipeline runner."""

    def __init__(
        self,
        args: OmniThinkRunnerArguments,
        lm_configs: OmniThinkLLMConfigs,
        rm,
    ):
        super().__init__(lm_configs=lm_configs)
        self.args = args
        self.lm_configs = lm_configs
        self.seed = args.seed

        self.retriever = Retriever(rm=rm, max_thread=1)
        self.lm = self.lm_configs.lm

    def run(
        self,
        topic,
        do_research=None,
        do_generate_outline=None,
        do_generate_article=None,
        do_polish_article=None,
        load_mind_map=True,
        save_mind_map=False,
    ):

        topic_name = topic.replace(" ", "_")
        manual_runs = [
            "manual_run_2025-05-04_00-05-12",
            "manual_run_2025-05-04_00-10-25",
            "manual_run_2025-05-04_00-20-05",
            "manual_run_2025-05-04_00-24-45",
            "manual_run_2025-05-04_00-24-47",
        ]

        # Path ie.: ~/output/apollo/gen_articles/SciWiki-100/domain/801_job
        if self.args.output_dir.split("/")[-1] not in manual_runs:
            print(
                f"Output directory {self.args.output_dir} is not in the list of manual runs."
            )
            return

        save_dir = os.path.join(
            self.args.output_dir,
            topic_name,
        )
        self.save_dir = save_dir

        if load_mind_map:
            mind_map = MindMap(
                retriever=self.retriever,
                gen_concept_lm=self.lm,
                depth=3,
                max_categories=3,
            )
            json_path = os.path.join(save_dir, "mindmap_d3_top_5.json")
            mind_map.load_map(json_path)
        else:
            mind_map = MindMap(
                retriever=self.retriever,
                gen_concept_lm=self.lm,
                depth=self.args.depth,
            )
            generator = mind_map.build_map(topic)
            for layer in generator:
                print(layer)

        if save_mind_map:
            mindmap_path = os.path.join(
                save_dir,
                f"mindmap_d{self.args.depth}_top_{self.args.search_top_k}.json",
            )
            if not os.path.exists(mindmap_path):
                print("Saving mindmap to", mindmap_path)
                with open(mindmap_path, "w", encoding="utf-8") as file:
                    mind_map.save_map(mind_map.root, mindmap_path)

        ogm = OutlineGenerationModule(self.lm)
        outline = ogm.generate_outline(
            topic=topic,
            mindmap=mind_map,
            save_dir=save_dir,
        )

        article_with_outline = Article.from_outline_str(
            topic=topic,
            outline_str=outline,
        )
        ag = ArticleGenerationModule(
            retriever=self.retriever,
            article_gen_lm=self.lm,
            retrieve_top_k=self.args.retrieve_top_k,
            max_thread_num=self.args.max_thread_num,
        )
        article = ag.generate_article(
            topic=topic,
            mindmap=mind_map,
            article_with_outline=article_with_outline,
            save_dir=save_dir,
        )
        
        ap = ArticlePolishingModule(
            article_gen_lm=self.lm,
            article_polish_lm=self.lm,
        )

        article_polished = ap.polish_article(
            topic=topic,
            draft_article=article,
            save_dir=save_dir,
        )


    def post_run(self):
            """
            Post-run operations, including:
            1. Dumping the run configuration.
            2. Dumping the LLM call history.
            """
            config_log = self.lm_configs.log()
            FileIOHelper.dump_json(
                config_log, os.path.join(self.save_dir, "run_config.json")
            )

            def custom_default(o):
                if hasattr(o, "to_dict"):
                    return o.to_dict()
                try:
                    return o.__dict__
                except AttributeError:
                    return str(o)

            llm_call_history = self.lm_configs.collect_and_reset_lm_history()
            with open(
                os.path.join(self.save_dir, "llm_call_history.jsonl"), "w"
            ) as f:
                for call in llm_call_history:
                    if "kwargs" in call:
                        call.pop("kwargs")
                    f.write(json.dumps(call, indent=4, default=custom_default) + "\n")
import os
import re
import json
import dspy
import numpy as np
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pipeline.omnithink.src.utils.ArticleTextProcessing import ArticleTextProcessing

import sys

sys.path.append("/home/toapantabarahonad/ds-agentic-topic-pages-gen")

import argparse
from pipeline.apollo.src import LLM
from pipeline.apollo.src import VectorRM, Retriever
from src.utils import (
    load_domains,
    format_args,
    setup_logging,
    get_logger,
    dump_json,
)

setup_logging()
logger = get_logger(__name__)


def eval_duplicate_sources(json_path, pipeline="storm"):

    with open(json_path, "r") as f:
        data = json.load(f)

    all_urls = []

    if pipeline == "storm":
        for entry in data:
            dlg_turns = entry.get("dlg_turns", [])
            for turn in dlg_turns:
                search_results = turn.get("search_results", [])
                for result in search_results:
                    url = result.get("url")
                    if url:
                        all_urls.append(url)
    elif pipeline == "omnithink":

        def collect_urls(node, all_urls):
            for info in node.get("info", []):
                url = info.get("url")
                if url:
                    all_urls.append(url)

            for child in node.get("children", {}).values():
                collect_urls(child, all_urls)

        collect_urls(data, all_urls)

    total_urls = len(all_urls)
    unique_urls = len(set(all_urls))
    duplicate_count = total_urls - unique_urls
    duplicate_pct = (duplicate_count / total_urls) * 100 if total_urls else 0

    print(f"Total URL entries found: {total_urls}")
    print(f"Unique URLs: {unique_urls}")
    print(f"Duplicate URLs: {duplicate_count} ({duplicate_pct:.2f}%)")

    return {
        "total_urls": total_urls,
        "unique_urls": unique_urls,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
    }


class ConceptGenerator(dspy.Module):
    """Extract information and generate a list of concepts."""

    def __init__(self, lm: dspy.LM):
        super().__init__()
        self.lm = lm
        self.concept_generator = dspy.Predict(GenConcept)

    def forward(self, infos: List[Dict]):
        snippets_list = []
        for info in infos:
            snippet = info.snippets
            snippets_list.extend(snippet)

        snippets_list_str = "\n".join(
            f"{index + 1}. {snippet}" for index, snippet in enumerate(snippets_list)
        )
        snippets_list_str = ArticleTextProcessing.limit_word_count_preserve_newline(
            snippets_list_str, 3000
        )

        with dspy.settings.context(lm=self.lm):
            concepts = self.concept_generator(info=snippets_list_str).concepts

        pattern = r"\d+\.\s*(.*)"
        matches = re.findall(pattern, concepts)
        concept_list = [match.strip() for match in matches]

        return concept_list


class ExtendConcept(dspy.Signature):
    """You are an analytical robot. I will provide you with a subject, the information I have searched about it, and our preliminary concept of it. I need you to generate a detailed, in-depth, and insightful report based on it, further exploring our initial ideas.

    First, break down the subject into several broad categories, then create corresponding search engine keywords for each category.

    Note: The new categories should not repeat the previous ones.

    Your output format should be as follows:
    -[Category 1]
    --{Keyword 1}
    --{Keyword 2}
    -[Category 2]
    --{Keyword 1}
    --{Keyword 2}"""

    info = dspy.InputField(
        prefix="The information you have collected from the webpage:", format=str
    )
    concept = dspy.InputField(
        prefix="The summary of the previous concepts:", format=str
    )
    category = dspy.InputField(
        prefix="The broader categories you need to further expand:", format=str
    )
    keywords = dspy.OutputField(format=str)


class GenConcept(dspy.Signature):
    """Please analyze, summarize, and evaluate the following webpage information.
    Think like a person, distill the core point of each piece of information, and synthesize them into a comprehensive opinion.
    Present your comprehensive opinion in the format of 1. 2. ..."""

    info = dspy.InputField(
        prefix="The webpage information you have collected:", format=str
    )
    concepts = dspy.OutputField(format=str)


class MindPoint:
    def __init__(
        self,
        retriever,
        lm: dspy.LM,
        root: bool = False,
        children: Optional[List["MindPoint"]] = None,
        concept: str = "",
        info: Optional[List[Dict]] = None,
        category: str = "",
    ):
        self.root = root
        self.category = category
        self.children = children if children is not None else {}
        self.concept = concept
        self.info = info if info is not None else []
        self.lm = lm
        self.retriever = retriever
        self.concept_generator = ConceptGenerator(lm=lm)

    def extend(self, max_categories=3, remaining_budget=None, debugging=False):
        extend_concept = dspy.Predict(ExtendConcept)
        with dspy.settings.context(lm=self.lm):
            info_str = "\n".join([str(i) for i in self.info])
            keywords = extend_concept(
                info=info_str,
                concept=self.concept,
                category=self.category,
            ).keywords
        categories = {}
        current_category = None

        for line in keywords.split("\n"):
            line = line.strip()
            if line.startswith("-[") and line.endswith("]"):
                current_category = line[2:-1]
                categories[current_category] = []
            elif line.startswith("- [") and line.endswith("]"):
                current_category = line[3:-1]
                categories[current_category] = []

            elif current_category is not None and line.startswith("--"):
                if "{" in line and "}" in line:
                    keyword = line[line.find("{") + 1 : line.find("}")].strip()
                else:
                    keyword = line[2:].strip()

                if keyword:
                    categories[current_category].append(keyword)

        if debugging:
            print(f"\n{'='*50}")
            print(f"Extending node: '{self.category}'")
            print(f"Generated {len(categories)} categories:")
            for i, (cat, keywords) in enumerate(categories.items()):
                print(f"  {i+1}. {cat} - {len(keywords)} keywords")

        # Apply limit if specified
        if max_categories and len(categories) > max_categories:
            limited_categories = dict(list(categories.items())[:max_categories])
            if debugging:
                print(f"Limited to {max_categories} categories")
        else:
            limited_categories = categories

        total_snippets = 0

        for category, keywords_list in limited_categories.items():
            # Check if we have budget before even retrieving
            if remaining_budget is not None and total_snippets >= remaining_budget:
                if debugging:
                    print(
                        f"Budget exhausted ({total_snippets}/{remaining_budget}), stopping"
                    )
                break

            # Retrieve information
            new_info = self.retriever(keywords_list)

            if new_info:
                snippets_count = sum(len(info.snippets) for info in new_info)

                # Check if adding these snippets would exceed budget
                if (
                    remaining_budget is not None
                    and total_snippets + snippets_count > remaining_budget
                ):
                    if debugging:
                        print(
                            f"Adding {snippets_count} snippets would exceed budget ({total_snippets + snippets_count} > {remaining_budget}), stopping"
                        )
                    break

                total_snippets += snippets_count

            # Only create the child if we're within budget
            new_concept = self.concept_generator.forward(new_info)
            new_node = MindPoint(
                concept=new_concept,
                info=new_info,
                lm=self.lm,
                retriever=self.retriever,
                category=category,
            )
            self.children[category] = new_node
            if debugging:
                print(f"  Total snippets for '{category}': {snippets_count}")

        if debugging:
            print(f"Total snippets retrieved for this node: {total_snippets}")
        return total_snippets


class MindMap:
    def __init__(
        self,
        retriever,
        gen_concept_lm: dspy.LM,
        depth: int,
        max_categories,
        workers: int = 5,
    ):
        self.retriever = retriever
        self.gen_concept_lm = gen_concept_lm
        self.depth = depth
        self.concept_generator = ConceptGenerator(lm=self.gen_concept_lm)
        self.root = None
        self.max_workers = workers
        self.max_categories = max_categories
        print("MindMap initialized")

    def build_map(self, topic: str, max_total_snippets=135, debugging=False):
        root_info = self.retriever(topic)
        root_concept = self.concept_generator(root_info)
        root = MindPoint(
            root=True,
            info=root_info,
            concept=root_concept,
            lm=self.gen_concept_lm,
            retriever=self.retriever,
            category=topic,
        )
        self.root = root

        total_snippets = sum(len(info.snippets) for info in root.info)
        current_level = [root]

        for count in range(self.depth):
            yield current_level

            if count == self.depth - 1 or total_snippets >= max_total_snippets:
                break

            next_level = []

            # Use a thread-safe counter
            import threading

            snippets_lock = threading.Lock()

            def process_node(node):
                nonlocal total_snippets

                # Check if we should continue
                with snippets_lock:
                    if total_snippets >= max_total_snippets:
                        return 0, []
                    remaining_budget = max_total_snippets - total_snippets

                # Extend the node
                snippets_added = node.extend(
                    max_categories=self.max_categories,
                    remaining_budget=remaining_budget,
                    debugging=debugging,
                )

                # Update total count thread-safely
                with snippets_lock:
                    total_snippets += snippets_added

                return snippets_added, list(node.children.values())

            # Process all nodes in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(process_node, node) for node in current_level
                ]

                for future in concurrent.futures.as_completed(futures):
                    snippets_added, children = future.result()
                    if children:
                        next_level.extend(children)

                    # Check if we've hit the limit
                    with snippets_lock:
                        if total_snippets >= max_total_snippets:
                            if debugging:
                                print(
                                    f"Total limit reached ({total_snippets}), cancelling remaining tasks"
                                )

                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            break

            current_level = next_level
            if debugging:
                print(f"Level {count + 1} complete. Total snippets: {total_snippets}")
        if debugging:
            print(f"Final total snippets: {total_snippets}")

    def build_map_works(self, topic: str, max_total_snippets=135, debugging=False):
        root_info = self.retriever(topic)
        root_concept = self.concept_generator(root_info)
        root = MindPoint(
            root=True,
            info=root_info,
            concept=root_concept,
            lm=self.gen_concept_lm,
            retriever=self.retriever,
            category=topic,
        )
        self.root = root

        total_snippets = sum(len(info.snippets) for info in root.info)
        current_level = [root]

        for count in range(self.depth):
            yield current_level

            if count == self.depth - 1 or total_snippets >= max_total_snippets:
                break

            next_level = []

            # Process nodes sequentially to maintain precise control
            for node in current_level:
                if total_snippets >= max_total_snippets:
                    if debugging:
                        print(
                            f"Total limit reached ({total_snippets}), stopping all expansions"
                        )
                    break

                remaining_budget = max_total_snippets - total_snippets
                snippets_added = node.extend(
                    max_categories=args.max_categories,
                    remaining_budget=remaining_budget,
                )
                total_snippets += snippets_added

                # Only add children that were actually created
                next_level.extend(node.children.values())

            current_level = next_level
            if debugging:
                print(f"Level {count + 1} complete. Total snippets: {total_snippets}")

        if debugging:
            print(f"Final total snippets: {total_snippets}")

    def recursive_extend(self, node: MindPoint, count: int):
        if count >= self.depth:
            return
        node.extend()
        count += 1

    def save_map(self, root: MindPoint, filename: str):
        def serialize_node(node: MindPoint):
            return {
                "category": node.category,
                "concept": node.concept,
                "children": {k: serialize_node(v) for k, v in node.children.items()},
                "info": [i.to_dict() for i in node.info],
            }

        mind_map_dict = serialize_node(root)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(mind_map_dict, f, ensure_ascii=False, indent=4)

    def load_map(self, filename: str):
        from pipeline.apollo.src.core.information import Information

        def deserialize_node(node_data):
            category = node_data["category"]
            concept = node_data["concept"]
            info = [Information.from_dict(i) for i in node_data["info"]]
            children_data = node_data["children"]

            node = MindPoint(
                concept=concept,
                info=info,
                lm=self.gen_concept_lm,
                retriever=self.retriever,
                category=category,
            )
            node.children = {k: deserialize_node(v) for k, v in children_data.items()}
            return node

        with open(filename, "r", encoding="utf-8") as f:
            mind_map_dict = json.load(f)

        self.root = deserialize_node(mind_map_dict)
        return self.root

    def export_categories_and_concepts(self) -> str:
        root = self.root
        output = []

        def traverse(node: MindPoint, indent=0):
            output.append(" " * indent + node.category)
            for concept in node.concept:
                output.append(" " * (indent + 2) + concept)
            for child in node.children.values():
                traverse(child, indent + 2)

        traverse(root)
        return "\n".join(output)

    def get_all_infos(self) -> List[Dict[str, any]]:
        """
        Get all unique info from the MindMap, ensuring unique URLs.
        """
        all_infos = []
        seen_urls = set()

        def traverse(node: MindPoint):
            if node.info:
                for info in node.info:
                    url = info.url
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_infos.append(info)
            for child in node.children.values():
                traverse(child)

        traverse(self.root)
        self.all_infos = all_infos
        return all_infos

    def prepare_table_for_retrieval(self):
        """
        Prepare collected snippets and URLs for retrieval by encoding the snippets using paraphrase-MiniLM-L6-v2.
        collected_urls and collected_snippets have corresponding indices.
        """
        self.encoder = SentenceTransformer(
            "/mnt/nas-alinlp/xizekun/huggingface_cache/all-MiniLM-L6-v2"
        )
        self.collected_urls = []
        self.collected_snippets = []
        seen_urls = set()

        for info in self.get_all_infos():
            url = info.url
            snippets = info.snippets
            if url and url not in seen_urls:
                seen_urls.add(url)
                for snippet in snippets:
                    self.collected_urls.append(url)
                    self.collected_snippets.append(snippet)

        self.encoded_snippets = self.encoder.encode(
            self.collected_snippets, show_progress_bar=True
        )

    def retrieve_information(
        self, queries: Union[List[str], str], search_top_k
    ) -> List[Dict[str, any]]:
        """
        Retrieve relevant information based on the given queries.
        Returns a list of dictionaries containing 'url' and 'snippets'.
        """
        selected_urls = []
        selected_snippets = []
        if type(queries) is str:
            queries = [queries]
        for query in queries:
            encoded_query = self.encoder.encode(query, show_progress_bar=False)
            sim = cosine_similarity([encoded_query], self.encoded_snippets)[0]
            sorted_indices = np.argsort(sim)
            for i in sorted_indices[-search_top_k:][::-1]:
                selected_urls.append(self.collected_urls[i])
                selected_snippets.append(self.collected_snippets[i])

        url_to_snippets = {}
        for url, snippet in zip(selected_urls, selected_snippets):
            if url not in url_to_snippets:
                url_to_snippets[url] = set()
            url_to_snippets[url].add(snippet)

        result = []
        for url, snippets in url_to_snippets.items():
            result.append({"url": url, "snippets": list(snippets)})

        return result

    def visualize_map_pyvis(self, root: MindPoint, output_file="mindmap.html"):
        """
        Create an interactive visualization of the mind map using pyvis.
        """
        from pyvis.network import Network
        from matplotlib import cm
        import matplotlib.colors as mcolors

        # Create the network
        net = Network(height="900px", width="100%", directed=True, notebook=False)

        node_ids = {}
        level_colors = {}

        # Generate a color map with up to 20 unique colors
        max_depth = self.depth + 1
        cmap = plt.get_cmap("tab20", max_depth)

        def get_color(level):
            if level not in level_colors:
                rgba = cmap(level)
                hex_color = mcolors.to_hex(rgba)
                level_colors[level] = hex_color
            return level_colors[level]

        def add_nodes_edges(node: MindPoint, parent_id=None, level=0):
            # Assign a unique ID per node category
            if node.category not in node_ids:
                node_ids[node.category] = len(node_ids) + 1
            node_id = node_ids[node.category]

            # Add the node with a color by level
            net.add_node(
                node_id,
                label=node.category,
                title=node.category,
                color=get_color(level),
            )

            if parent_id is not None:
                net.add_edge(parent_id, node_id)

            for child in node.children.values():
                add_nodes_edges(child, node_id, level + 1)

        # Build network from root
        add_nodes_edges(root)

        # Save as interactive HTML
        net.save_graph(output_file)
        print(f"Interactive mind map saved to: {output_file}")


def mk_mindmap(args, lm, retriever):

    # lm = LLM(
    #     model="gpt-4o-mini",
    #     temperature=1,
    #     max_tokens=512,
    #     cache=False,
    # )

    # retriever = _setup_retrieval(args)
    # # results = retriever("Ensemble learning")
    # # retriever.print_results(results)

    mind_map = MindMap(
        retriever, lm, depth=args.depth, max_categories=args.max_categories
    )
    topic_name = args.topic.replace(" ", "_")

    levels = list(
        mind_map.build_map(
            args.topic,
            max_total_snippets=args.max_total_snippets,
            debugging=args.debugging,
        )
    )

    if args.debugging:
        print("\nFinal Level Structure:")
        for i, level in enumerate(levels):
            print(
                f"Level {i}: {len(level)} nodes - {[node.category for node in level]}"
            )

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        total_nodes = count_nodes(mind_map.root)

        if args.debugging:
            print(f"\nTotal nodes in tree: {total_nodes}")

    save_dir = f"{args.output_dir}/{args.domain}/{topic_name}/{args.jobid or ''}"
    os.makedirs(save_dir, exist_ok=True)

    if mind_map.root:
        output_file_html = f"{save_dir}/mindmap_d{args.depth}_top_{args.top_k}.html"
        mind_map.visualize_map_pyvis(mind_map.root, output_file_html)

        json_file = f"{save_dir}/mindmap_d{args.depth}_top_{args.top_k}.json"
        mind_map.save_map(mind_map.root, json_file)
        print(f"Mind map data saved to {json_file}")

        result = eval_duplicate_sources(json_file, pipeline="omnithink")

        return result


def _setup_retrieval(args):
    rm = VectorRM(
        collection_name=args.domain,
        embedding_model=args.embedding_model,
        device=args.device,
        k=args.top_k,
        seed=args.seed,
    )
    rm.set_filter_by(args.topic)
    rm.k = args.top_k

    return Retriever(rm=rm, max_thread=1)


def main(args):

    all_results = []
    domains: dict = load_domains()

    lm = LLM(
        model="gpt-4o-mini",
        temperature=1,
        max_tokens=512,
        cache=False,
    )

    retriever = _setup_retrieval(args)

    for i, (domain, topics) in enumerate(tqdm(domains.items(), desc="Domain")):
        logger.info(f"\n** Domain: {domain} **")

        retriever.rm.collection_name = domain
        retriever.rm.init_docker_qdrant()

        for topic in tqdm(topics, desc="Generating articles"):
            topic_name = topic.replace(" ", "_")

            print(f"\n=== Topic: {topic_name} ===")

            args.topic = topic
            args.domain = domain

            retriever.rm.set_filter_by(topic)

            try:
                result = mk_mindmap(args, lm, retriever)
                result.update(
                    {
                        "domain": domain,
                        "topic": topic_name,
                    }
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {topic_name}: {e}")
                continue

        retriever.rm.cleanup()

    if all_results:
        total_urls = sum(r["total_urls"] for r in all_results)
        unique_urls = sum(r["unique_urls"] for r in all_results)
        duplicate_count = sum(r["duplicate_count"] for r in all_results)
        overall_dup_pct = (duplicate_count / total_urls) * 100 if total_urls else 0

        summary = {
            "pipeline": args.pipeline,
            "num_topics": len(all_results),
            "total_urls": total_urls,
            "unique_urls": unique_urls,
            "duplicate_count": duplicate_count,
            "overall_duplicate_pct": overall_dup_pct,
        }

        # write per-topic and summary to disk
        out_summary_path = os.path.join(
            args.result_output_dir,
            args.jobid or "",
            "summary.json",
        )
        out_details_path = os.path.join(
            args.result_output_dir,
            args.jobid or "",
            "per_topic.json",
        )
        dump_json(summary, out_summary_path)
        dump_json(all_results, out_details_path)

        print(f"\nOverall summary results: {out_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobid",
        required=False,
    )
    args, unknown = parser.parse_known_args()
    args.seed = 42
    args.device = "cuda"
    args.embedding_model = "Snowflake/snowflake-arctic-embed-m-v2.0"
    args.output_dir = "/home/toapantabarahonad/ds-agentic-topic-pages-gen/output/omnithink/gen_articles/SciWiki-100"
    args.result_output_dir = "/home/toapantabarahonad/ds-agentic-topic-pages-gen/output/omnithink/metrics/sources_eval_results/"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_output_dir, exist_ok=True)

    args.domain = "ComputerScience"

    # args.topic = "Ensemble learning"
    args.topic = "Linear discriminant analysis"
    # args.topic = "Network time protocol"

    args.pipeline = "omnithink"
    args.top_k = 5
    args.depth = 3
    args.max_categories = 3
    args.max_total_snippets = 135  # This is a cap in theory it will always be below this threshold but added as safeguard
    args.debugging = False
    main(args)

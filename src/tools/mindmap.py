import concurrent.futures
import os
import re
import json
import dspy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.ArticleTextProcessing import ArticleTextProcessing


script_dir = os.path.dirname(os.path.abspath(__file__))


import json


def print_duplicate_summary(json_path, pipeline="storm"):

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

    def extend(self, max_categories=3, remaining_budget=None):
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

        # Debug prints
        print(f"\n{'='*50}")
        print(f"Extending node: '{self.category}'")
        print(f"Generated {len(categories)} categories:")
        for i, (cat, keywords) in enumerate(categories.items()):
            print(f"  {i+1}. {cat} - {len(keywords)} keywords")

        # Apply limit if specified
        if max_categories and len(categories) > max_categories:
            limited_categories = dict(list(categories.items())[:max_categories])
            print(f"Limited to {max_categories} categories")
        else:
            limited_categories = categories
    
        total_snippets = 0
    
        for category, keywords_list in limited_categories.items():
            # Check if we have budget before even retrieving
            if remaining_budget is not None and total_snippets >= remaining_budget:
                print(f"Budget exhausted ({total_snippets}/{remaining_budget}), stopping")
                break
                
            # Retrieve information
            new_info = self.retriever(keywords_list)
            
            if new_info:
                snippets_count = sum(len(info.snippets) for info in new_info)
                
                # Check if adding these snippets would exceed budget
                if remaining_budget is not None and total_snippets + snippets_count > remaining_budget:
                    print(f"Adding {snippets_count} snippets would exceed budget ({total_snippets + snippets_count} > {remaining_budget}), stopping")
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
            print(f"  Total snippets for '{category}': {snippets_count}")
        
        print(f"Total snippets retrieved for this node: {total_snippets}")
        return total_snippets

    def extend_debug_claude(self, max_categories=None):
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

        # Debug prints
        print(f"\n{'='*50}")
        print(f"Extending node: '{self.category}'")
        print(f"Generated {len(categories)} categories:")
        for i, (cat, keywords) in enumerate(categories.items()):
            print(f"  {i+1}. {cat} - {len(keywords)} keywords")

        # Apply limit if specified
        if max_categories and len(categories) > max_categories:
            limited_categories = dict(list(categories.items())[:max_categories])
            print(f"Limited to {max_categories} categories")
        else:
            limited_categories = categories

        total_snippets = 0
        for category, keywords_list in limited_categories.items():
            print(f"  Retrieving for '{category}' with keywords: {keywords_list}")
            new_info = self.retriever(keywords_list)

            if not new_info:
                print(f"Warning: No information retrieved for category: {category}")
            else:
                # More detailed debugging
                print(f"  Retrieved {len(new_info)} info objects")
                snippets_count = 0
                for i, info in enumerate(new_info):
                    if hasattr(info, "snippets"):
                        snippets_count += len(info.snippets)
                        print(f"    Info {i}: {len(info.snippets)} snippets")
                    elif isinstance(info, dict) and "snippets" in info:
                        snippets_count += len(info["snippets"])
                        print(f"    Info {i}: {len(info['snippets'])} snippets")
                total_snippets += snippets_count
                print(f"  Total snippets for '{category}': {snippets_count}")

            new_concept = self.concept_generator.forward(new_info)
            new_node = MindPoint(
                concept=new_concept,
                info=new_info,
                lm=self.lm,
                retriever=self.retriever,
                category=category,
            )
            self.children[category] = new_node

        print(f"Total snippets retrieved for this node: {total_snippets}")
        print(f"{'='*50}\n")

    def extend_(self):
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

        for category, keywords_list in categories.items():
            new_info = self.retriever(keywords_list)
            if not new_info:
                print(f"Warning: No information retrieved for category: {category}")
            new_concept = self.concept_generator.forward(new_info)
            new_node = MindPoint(
                concept=new_concept,
                info=new_info,
                lm=self.lm,
                retriever=self.retriever,
                category=category,
            )
            self.children[category] = new_node


class MindMap:
    def __init__(
        self,
        retriever,
        gen_concept_lm: dspy.LM,
        depth: int,
        workers: int = 5,
    ):
        self.retriever = retriever
        self.gen_concept_lm = gen_concept_lm
        self.depth = depth
        self.concept_generator = ConceptGenerator(lm=self.gen_concept_lm)
        self.root = None
        self.max_workers = workers
        print("MindMap initialized")

    def build_map(self, topic: str, max_total_snippets=135):
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
                    print(f"Total limit reached ({total_snippets}), stopping all expansions")
                    break
                    
                remaining_budget = max_total_snippets - total_snippets
                snippets_added = node.extend(max_categories=args.max_categories, remaining_budget=remaining_budget)
                total_snippets += snippets_added
                
                # Only add children that were actually created
                next_level.extend(node.children.values())

            current_level = next_level
            print(f"Level {count + 1} complete. Total snippets: {total_snippets}")

        print(f"Final total snippets: {total_snippets}")
        
    def build_map__(self, topic: str, max_total_snippets=180):
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
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Calculate remaining budget for each node
                remaining_budget = max_total_snippets - total_snippets
                futures = {
                    executor.submit(node.extend, remaining_budget=remaining_budget): node 
                    for node in current_level
                }

                for future in concurrent.futures.as_completed(futures):
                    node = futures[future]
                    snippets_added = future.result()
                    total_snippets += snippets_added
                    
                    # Only add children that were actually created
                    next_level.extend(node.children.values())
                    
                    if total_snippets >= max_total_snippets:
                        print(f"Total limit reached ({total_snippets}), stopping all expansions")
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

            current_level = next_level
            print(f"Level {count + 1} complete. Total snippets: {total_snippets}")

        print(f"Final total snippets: {total_snippets}")

    def build_map_omni(self, topic: str):
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

        current_level = [root]

        for count in range(self.depth):
            next_level = []

            yield current_level
            if count == self.depth - 1:  # Check if it's the last layer
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(node.extend): node for node in current_level}

                for future in concurrent.futures.as_completed(futures):
                    node = futures[future]
                    # Assuming extend populates children.
                    next_level.extend(node.children.values())

            # yield current_level
            current_level = next_level

    def build_map_(self, topic: str):
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
        current_level = [root]

        for count in range(self.depth):
            next_level = []
            # Yield the current level before processing children.
            yield current_level
            # If it's the last layer, break out of the loop.
            if count == self.depth - 1:
                break

            for node in current_level:
                node.extend()
                next_level.extend(node.children.values())

            # Optionally, yield the current level after processing (if that's intended).
            yield current_level

            # Move to the next level.
            current_level = next_level

        # for count in range(self.depth):
        #     next_level = []

        #     yield current_level
        #     if count == self.depth - 1:  # Check if it's the last layer
        #         break

        #     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        #         futures = {executor.submit(node.extend): node for node in current_level}

        #         for future in concurrent.futures.as_completed(futures):
        #             node = futures[future]
        #             # Assuming extend populates children.
        #             next_level.extend(node.children.values())

        #     yield current_level
        #     current_level = next_level

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

    def visualize_map(self, root: MindPoint, output_file="mindmap.png"):
        G = nx.DiGraph()

        def add_edges(node: MindPoint, parent=None):
            if parent is not None:
                G.add_edge(parent, node.category)
            for child in node.children.values():
                add_edges(child, node.category)

        add_edges(root)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
        plt.title("MindMap Visualization", fontsize=15)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Mind map visualization saved to {output_file}")


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

    lm = LLM(
        model="gpt-4o-mini",
        temperature=1,
        max_tokens=512,
        cache=False,
    )

    retriever = _setup_retrieval(args)
    # results = retriever("Ensemble learning")
    # retriever.print_results(results)

    mind_map = MindMap(retriever, lm, depth=args.depth)
    topic_name = args.topic.replace(" ", "_")

    levels = list(mind_map.build_map(args.topic), max_total_snippets=args.max_total_snippets)

    # Print level structure
    print("\nFinal Level Structure:")
    for i, level in enumerate(levels):
        print(f"Level {i}: {len(level)} nodes - {[node.category for node in level]}")

    # Count total nodes
    def count_nodes(node):
        count = 1
        for child in node.children.values():
            count += count_nodes(child)
        return count

    total_nodes = count_nodes(mind_map.root)
    print(f"\nTotal nodes in tree: {total_nodes}")

    save_dir = f"{args.output_dir}/{topic_name}"
    os.makedirs(save_dir, exist_ok=True)

    if mind_map.root:
        # output_file_png = f"{save_dir}/mindmap.png"
        # mind_map.visualize_map(mind_map.root, output_file_png)

        output_file_html = f"{save_dir}/mindmap_d{args.depth}_top_{args.top_k}.html"
        mind_map.visualize_map_pyvis(mind_map.root, output_file_html)

        json_file = f"{save_dir}/mindmap_d{args.depth}_top_{args.top_k}.json"
        mind_map.save_map(mind_map.root, json_file)
        print(f"Mind map data saved to {json_file}")

        print_duplicate_summary(json_file, pipeline="omnithink")


if __name__ == "__main__":
    import dspy
    import argparse
    from pipeline.apollo.src import LLM
    from pipeline.apollo.src import VectorRM, Retriever

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42)
    args, unknown = parser.parse_known_args()
    args.output_dir = (
        "/home/toapantabarahonad/ds-agentic-topic-pages-gen/pipeline/omnithink/tmp"
    )

    args.domain = "ComputerScience"
    # args.topic = "Ensemble learning"
    args.topic = "Linear discriminant analysis"
    # args.topic = "Network time protocol"

    args.embedding_model = "Snowflake/snowflake-arctic-embed-m-v2.0"
    args.device = "cuda"
    args.seed = 42
    args.top_k = 5
    args.depth = 3
    args.max_categories = 3
    args.max_total_snippets = 135
    main(args)

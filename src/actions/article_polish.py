import copy
import dspy
import os
from ..utils.ArticleTextProcessing import ArticleTextProcessing
from ..dataclass.Article import Article

# This code is originally sourced from Repository STORM
# URL: [https://github.com/stanford-oval/storm]
class ArticlePolishingModule:
    """
    The interface for article generation stage. Given topic, collected information from
    knowledge curation stage, generated outline from outline generation stage.
    """

    def __init__(
        self,
        article_gen_lm: dspy.LM,
        article_polish_lm: dspy.LM,
    ):
        self.article_gen_lm = article_gen_lm
        self.article_polish_lm = article_polish_lm

        self.polish_page = PolishPageModule(
            write_lead_engine=self.article_gen_lm, polish_engine=self.article_polish_lm
        )

    def polish_article(
        self,
        topic: str,
        draft_article,
        remove_duplicate: bool = False,
        save_dir: str = None,
    ):
        """
        Polish article.

        Args:
            topic (str): The topic of the article.
            draft_article (StormArticle): The draft article.
            remove_duplicate (bool): Whether to use one additional LM call to remove duplicates from the article.
        """
        # if not remove_duplicate:
        #     return draft_article

        article_text = draft_article.to_string()
        remove_duplicate = True
        polish_result = self.polish_page(
            topic=topic, draft_page=article_text, polish_whole_page=remove_duplicate
        )

        polished_article = polish_result.page

        polished_article_dict = ArticleTextProcessing.parse_article_into_dict(
            polished_article
        )
        polished_article: Article = copy.deepcopy(draft_article)
        polished_article.insert_or_create_section(article_dict=polished_article_dict)
        polished_article.post_processing()

        # Save Polished Article
        article_polished_path = os.path.join(
            save_dir,
            f"omnithink_gen_article_polished.md",
        )

        # Save References
        article_references_path = os.path.join(
            save_dir,
            f"url_to_info_polished.json",
        )
        polished_article.dump_reference_to_file(article_references_path)
        polished_article.dump_article_as_plain_text(article_polished_path)

        return polished_article


class PolishPage(dspy.Signature):
    """You are a faithful text editor that is good at finding repeated information in the article and deleting them to make sure there is no repetition in the article. You won't delete any non-repeated part in the article. You will keep the inline citations and article structure (indicated by "#", "##", etc.) appropriately. Do your job for the following article."""

    article = dspy.InputField(prefix="The article you need to polish:\n", format=str)
    page = dspy.OutputField(prefix="Your revised article:\n", format=str)


class PolishPageModule(dspy.Module):
    def __init__(
        self,
        write_lead_engine: dspy.LM,
        polish_engine: dspy.LM,
    ):
        super().__init__()
        self.write_lead_engine = write_lead_engine
        self.polish_engine = polish_engine
        self.polish_page = dspy.Predict(PolishPage)

    def forward(self, topic: str, draft_page: str, polish_whole_page: bool = True):

        with dspy.settings.context(lm=self.polish_engine):
            page = self.polish_page(article=draft_page).page

        return dspy.Prediction(page=page)

from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List, Optional
import click
from oaklib import get_adapter
from agent_tutorial.oak_agent import search_cl

class TextAnnotation(BaseModel):
    """
    A text annotation is a span of text and the cl ID and label for the cell type it mentions.
    Use `text` for the source text, and `cl_id` and `cl_label` for the cl ID and label of the cell type in the ontology.
    """
    text: str
    cl_id: Optional[str] = None
    cl_label: Optional[str] = None

class TextAnnotationResult(BaseModel):
    annotations: List[TextAnnotation]

annotator_agent = Agent(
    #'claude-3-7-sonnet-latest',
    'openai:gpt-4o',
    system_prompt="""
    You will be provided with a table where each row represents a cell type.
    Your goal is to map text in each row to terms from the cell ontology.
    Return the as a list of annotations.
    Be sure to include all spans mentioning cell types;
    convert all plurals to singular before searching;
    if you cannot find a cl ID, then you should still return a TextAnnotation, just leave
    the cl_id field empty.

    However, before giving up you should be sure to try different combinations of
    synonyms with the `search_cl` tool. When you try searching synonyms, please also
    try substituting some terms in the span with common synonyms of those terms.  
    Also try converting between the forms 'X Y' and 'Y of X' where X is a tissue 
    or anatomical structure and Y is a cell type.
    """,
    tools=[search_cl],
    result_type=TextAnnotationResult,  
)
DEFAULT_TEXT = 'C_goblet	Colon goblet cells	Colonic goblet cells; Mucus-producing cells; GC	Crypt-resident goblet cells; Intercrypt goblet cells; Early goblet cells'

@click.command()
@click.argument('text', default=DEFAULT_TEXT)
def main(text: str):
    """Run the annotator agent on the given text."""
    result = annotator_agent.run_sync(text)
    print("## Result:")
    for a in result.data.annotations:
        print(f"  {a.text} ==> {a.cl_id} {a.cl_label}")

if __name__ == "__main__":
    main()

from pydantic import BaseModel
from pydantic_ai import Agent
from typing import List, Optional
import click
from oaklib import get_adapter
from agent_tutorial.oak_agent import search_uberon

class TextAnnotation(BaseModel):
    """
    A text annotation is a span of text and the UBERON ID and label for the anatomical structure it mentions.
    Use `text` for the source text, and `uberon_id` and `uberon_label` for the UBERON ID and label of the anatomical structure in the ontology.
    """
    text: str
    uberon_id: Optional[str] = None
    uberon_label: Optional[str] = None

class TextAnnotationResult(BaseModel):
    annotations: List[TextAnnotation]

annotator_agent = Agent(  
    'claude-3-7-sonnet-latest',
    system_prompt="""
    Extract all uberon terms from the text. Return the as a list of annotations.
    Be sure to include all spans mentioning anatomical structures; if you cannot
    find a UBERON ID, then you should still return a TextAnnotation, just leave
    the uberon_id field empty.

    However, before giving up you should be sure to try different combinations of
    synonyms with the `search_uberon` tool.
    """,
    tools=[search_uberon],
    result_type=TextAnnotationResult,  
)
DEFAULT_TEXT = 'The (hippocampal) CA1 is located in the forebrain, it projects to the amygdala, subiculum, and the entorhinal cortex'

@click.command()
@click.argument('text', default=DEFAULT_TEXT)
def main(text: str):
    """Run the annotator agent on the given text."""
    result = annotator_agent.run_sync(text)
    print("## Result:")
    for a in result.data.annotations:
        print(f"  {a.text} ==> {a.uberon_id} {a.uberon_label}")

if __name__ == "__main__":
    main()

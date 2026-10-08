from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for  favourite teams.


Terrorism is a big danger to peace  , creates fear in cities and villages. When such attacks happen, they leave behind that pain. To fight terror t security forces, and support from people who care about peace  safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)


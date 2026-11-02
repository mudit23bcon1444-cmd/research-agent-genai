import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import scrape_url, web_search

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable is missing. "
        "Please create a .env file and set GROQ_API_KEY=gsk_..."
    )

if not groq_model:
    raise ValueError(
        "GROQ_MODEL environment variable is missing. "
        "Please set GROQ_MODEL in your .env file (e.g. openai/gpt-oss-120b or llama-3.3-70b-versatile)."
    )

# Shared ChatGroq LLM instance
llm = ChatGroq(
    model=groq_model,
    temperature=0,
)


def build_search_agent():
    """Builds and returns the search agent equipped with web_search."""
    return create_agent(
        llm,
        tools=[web_search],
        system_prompt=(
            "You are a web search assistant. Given a topic, invoke web_search once with a targeted query "
            "to find recent, high-quality sources. Once the search results return, immediately format and "
            "return the titles, URLs, and key snippets without additional searches."
        ),
    )


def build_reader_agent():
    """Builds and returns the reader agent equipped with scrape_url."""
    return create_agent(
        llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a research reader. Given search results, select the single most relevant and authoritative URL. "
            "Call the scrape_url tool ONCE with that URL (as the 'url' parameter). After receiving the scraped content, "
            "do NOT call any tool again. Immediately provide a detailed factual summary of what you learned from the page."
        ),
    )


# Writer Prompt & Chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. You produce structured, authoritative, and "
        "comprehensive research reports based strictly on provided research data. "
        "Do not invent information or sources not present in the research data."
    ),
    (
        "human",
        "Research Topic: {topic}\n\n"
        "Research Data:\n"
        "{research}\n\n"
        "Write a detailed research report following this exact structure:\n"
        "# {topic}\n\n"
        "## Introduction\n"
        "Provide context, background, and significance of the topic.\n\n"
        "## Key Findings\n"
        "Detail at least 3 distinct, well-explained key findings or insights with supporting evidence.\n\n"
        "## Conclusion\n"
        "Summarize the main takeaways, impact, and future outlook.\n\n"
        "## Sources\n"
        "List all unique source URLs present in the research data. Only cite URLs provided in the research."
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()


# Critic Prompt & Chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp, rigorous, and honest critic evaluating research reports for factual "
        "depth, clarity, structure, and balance. Always follow the required format strictly."
    ),
    (
        "human",
        "Please review the following research report:\n\n"
        "{report}\n\n"
        "Provide your evaluation using strictly the following format:\n\n"
        "Score: X/10\n\n"
        "Strengths:\n"
        "- [Point 1]\n"
        "- [Point 2]\n\n"
        "Areas to Improve:\n"
        "- [Point 1]\n"
        "- [Point 2]\n\n"
        "One-line verdict:\n"
        "[Single sentence summary verdict]"
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()

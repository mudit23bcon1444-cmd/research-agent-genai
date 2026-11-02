import sys
from rich.console import Console
from rich.panel import Panel

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

console = Console()


def run_research_pipeline(topic: str) -> dict:
    """
    Executes the 4-stage research pipeline following the reference repository structure:
      1. Search Agent: Finds recent, reliable, and detailed information.
      2. Reader Agent: Scrapes the single most relevant URL from the top 800 chars of search results.
      3. Writer Chain: Synthesizes search results and scraped content into a report.
      4. Critic Chain: Evaluates the report and outputs a score and critique.
    """
    state = {}

    # Stage 1: Search Agent
    console.print(Panel(f"[bold cyan]STAGE 1: SEARCH AGENT[/bold cyan]\nSearching for: {topic}", expand=False))
    search_agent = build_search_agent()
    search_prompt = f"Find recent, reliable and detailed information about: {topic}"
    search_res = search_agent.invoke({"messages": [{"role": "user", "content": search_prompt}]})
    search_results = search_res["messages"][-1].content
    state["search_results"] = search_results
    console.print("[dim]" + search_results + "[/dim]\n")

    # Stage 2: Reader Agent
    console.print(Panel("[bold green]STAGE 2: READER AGENT[/bold green]\nSelecting URL and scraping content...", expand=False))
    reader_agent = build_reader_agent()
    reader_prompt = (
        f"Based on the following search results about '{topic}', pick the most relevant URL "
        f"and scrape it for deeper content\n\nSearch Results:\n{state['search_results'][:800]}"
    )
    reader_res = reader_agent.invoke({"messages": [{"role": "user", "content": reader_prompt}]})
    scraped_content = reader_res["messages"][-1].content
    state["scraped_content"] = scraped_content
    console.print("[dim]" + scraped_content + "[/dim]\n")

    # Stage 3: Writer Chain
    console.print(Panel("[bold magenta]STAGE 3: WRITER CHAIN[/bold magenta]\nSynthesizing findings into structured report...", expand=False))
    combined_research = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    report = writer_chain.invoke({"topic": topic, "research": combined_research})
    state["report"] = report
    console.print(report + "\n")

    # Stage 4: Critic Chain
    console.print(Panel("[bold yellow]STAGE 4: CRITIC CHAIN[/bold yellow]\nEvaluating report...", expand=False))
    feedback = critic_chain.invoke({"report": state["report"]})
    state["feedback"] = feedback
    console.print(feedback + "\n")

    return state


if __name__ == "__main__":
    console.print("[bold cyan]ResearchMind: Multi-Agent Research System[/bold cyan]\n")
    if len(sys.argv) > 1:
        topic_input = " ".join(sys.argv[1:])
    else:
        try:
            topic_input = input("Enter a research topic: ").strip()
        except (EOFError, KeyboardInterrupt):
            topic_input = ""

    if not topic_input:
        topic_input = "Agentic AI workflows in 2026"
        console.print(f"[yellow]No topic entered. Using default: '{topic_input}'[/yellow]\n")

    run_research_pipeline(topic_input)

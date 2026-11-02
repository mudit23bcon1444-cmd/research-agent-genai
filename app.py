import time
import streamlit as st

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom Styling (Dark theme, Orange accent, Google Fonts)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300..800;1,9..40,300..800&family=Syne:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    .hero-container {
        padding: 1.5rem 0 2rem 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ff7a45 0%, #ff5722 50%, #ffa940 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    /* Step Card Styling */
    .step-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: all 0.25s ease;
    }

    .step-card.waiting {
        border-left: 4px solid #484f58;
        opacity: 0.7;
    }

    .step-card.running {
        border-left: 4px solid #ff7a45;
        background: linear-gradient(90deg, rgba(255,122,69,0.08) 0%, rgba(22,27,34,1) 100%);
        box-shadow: 0 0 15px rgba(255, 122, 69, 0.2);
    }

    .step-card.done {
        border-left: 4px solid #2ea043;
        background: linear-gradient(90deg, rgba(46,160,67,0.06) 0%, rgba(22,27,34,1) 100%);
    }

    .step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }

    .step-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.02rem;
        font-weight: 700;
        color: #f0f6fc;
    }

    .step-badge {
        font-family: 'DM Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        letter-spacing: 0.05em;
    }

    .badge-waiting {
        background-color: #21262d;
        color: #8b949e;
    }

    .badge-running {
        background-color: rgba(255, 122, 69, 0.2);
        color: #ff7a45;
        border: 1px solid rgba(255, 122, 69, 0.4);
    }

    .badge-done {
        background-color: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
    }

    .step-desc {
        color: #8b949e;
        font-size: 0.88rem;
        margin: 0;
    }

    /* Critic Panel */
    .critic-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-top: 4px solid #ff7a45;
        border-radius: 12px;
        padding: 1.4rem;
        font-family: 'DM Sans', sans-serif;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #ff7a45 0%, #ff5722 100%);
        color: #ffffff;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.2rem;
        font-size: 1rem;
        letter-spacing: 0.02em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #ff8c5a 0%, #ff6b35 100%);
        box-shadow: 0 4px 14px rgba(255, 87, 34, 0.35);
        color: #ffffff;
    }

    /* Example Chips */
    .chip-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }

    .footer-text {
        color: #6e7681;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #21262d;
        font-family: 'DM Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Step Card Helper Component
# -----------------------------------------------------------------------------
def step_card(num: int, title: str, state: str, desc: str):
    """
    Renders an animated status card for a pipeline stage.
    state: 'waiting' | 'running' | 'done'
    """
    state_labels = {
        "waiting": "Waiting",
        "running": "Running...",
        "done": "Completed",
    }
    badge_label = state_labels.get(state, "Waiting")

    card_html = f"""
    <div class="step-card {state}">
        <div class="step-header">
            <span class="step-title">0{num}. {title}</span>
            <span class="step-badge badge-{state}">{badge_label}</span>
        </div>
        <p class="step-desc">{desc}</p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = {}

if "running" not in st.session_state:
    st.session_state.running = False

if "done" not in st.session_state:
    st.session_state.done = False

if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


# -----------------------------------------------------------------------------
# Hero Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">ResearchMind · AI Research Agent</h1>
        <p class="hero-subtitle">
            Autonomous multi-agent research pipeline powered by Groq, LangGraph, DuckDuckGo & BeautifulSoup
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Layout: 3 Columns (5 : 0.5 : 4)
# -----------------------------------------------------------------------------
col_left, col_mid, col_right = st.columns([5, 0.5, 4])


# Compute Stage States for Cards
def get_card_state(step_key: str, prev_key: str = None) -> str:
    if st.session_state.done:
        return "done"
    if not st.session_state.running:
        return "waiting"
    if step_key in st.session_state.results:
        return "done"
    if prev_key is None or prev_key in st.session_state.results:
        return "running"
    return "waiting"


search_state = get_card_state("search", None)
reader_state = get_card_state("reader", "search")
writer_state = get_card_state("writer", "reader")
critic_state = get_card_state("critic", "writer")


with col_left:
    st.markdown("### 🔎 Research Topic")
    topic = st.text_input(
        label="Topic Input",
        key="topic_input",
        placeholder="e.g. Advancements in Quantum Computing 2026",
        label_visibility="collapsed",
    )

    st.markdown('<div class="chip-label">Quick suggestions:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(3)
    suggestions = [
        "Quantum Computing in 2026",
        "Autonomous AI Agents in Healthcare",
        "Solid State Batteries Breakthroughs",
    ]
    for i, sug in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(sug, key=f"chip_{i}", use_container_width=True):
                st.session_state.topic_input = sug
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    start_clicked = st.button("🚀 Start Research Pipeline", use_container_width=True)

    if start_clicked:
        clean_topic = st.session_state.get("topic_input", "").strip()
        if not clean_topic:
            st.warning("Please enter a research topic or choose a suggestion above.")
        else:
            st.session_state.results = {}
            st.session_state.running = True
            st.session_state.done = False
            st.rerun()


with col_right:
    st.markdown("### ⚡ Pipeline Stages")
    step_card(1, "Search Agent", search_state, "Queries DuckDuckGo via ddgs for reliable titles, URLs & snippets")
    step_card(2, "Reader Agent", reader_state, "Picks top URL and extracts clean webpage body via BeautifulSoup")
    step_card(3, "Writer Chain", writer_state, "LCEL chain synthesizes findings into a comprehensive structured report")
    step_card(4, "Critic Chain", critic_state, "Scores report (X/10), identifies strengths, and highlights improvements")


# -----------------------------------------------------------------------------
# Pipeline Execution Loop (Step-by-Step with Spinners & UI Updates)
# -----------------------------------------------------------------------------
if st.session_state.running and not st.session_state.done:
    current_topic = st.session_state.get("topic_input", "").strip()

    # Step 1: Search Agent
    if "search" not in st.session_state.results:
        with st.spinner("Search Agent: Searching DuckDuckGo for recent sources..."):
            try:
                search_agent = build_search_agent()
                prompt = f"Find recent, reliable and detailed information about: {current_topic}"
                res = search_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                st.session_state.results["search"] = res["messages"][-1].content
            except Exception as e:
                st.error(f"Search Agent encountered an error: {e}")
                st.session_state.running = False
                st.stop()
        st.rerun()

    # Step 2: Reader Agent
    elif "reader" not in st.session_state.results:
        with st.spinner("Reader Agent: Selecting best URL and scraping full content..."):
            try:
                reader_agent = build_reader_agent()
                prompt = (
                    f"Based on the following search results about '{current_topic}', pick the most relevant URL "
                    f"and scrape it for deeper content\n\nSearch Results:\n{st.session_state.results['search'][:800]}"
                )
                res = reader_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
                st.session_state.results["reader"] = res["messages"][-1].content
            except Exception as e:
                st.error(f"Reader Agent encountered an error: {e}")
                st.session_state.running = False
                st.stop()
        st.rerun()

    # Step 3: Writer Chain
    elif "writer" not in st.session_state.results:
        with st.spinner("Writer Chain: Synthesizing findings into structured report..."):
            try:
                combined_research = (
                    f"SEARCH RESULTS:\n{st.session_state.results['search']}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n{st.session_state.results['reader']}"
                )
                report = writer_chain.invoke({
                    "topic": current_topic,
                    "research": combined_research
                })
                st.session_state.results["writer"] = report
            except Exception as e:
                st.error(f"Writer Chain encountered an error: {e}")
                st.session_state.running = False
                st.stop()
        st.rerun()

    # Step 4: Critic Chain
    elif "critic" not in st.session_state.results:
        with st.spinner("Critic Chain: Reviewing report and producing score..."):
            try:
                feedback = critic_chain.invoke({
                    "report": st.session_state.results["writer"]
                })
                st.session_state.results["critic"] = feedback
                st.session_state.running = False
                st.session_state.done = True
            except Exception as e:
                st.error(f"Critic Chain encountered an error: {e}")
                st.session_state.running = False
                st.stop()
        st.rerun()


# -----------------------------------------------------------------------------
# Results Display Section
# -----------------------------------------------------------------------------
if st.session_state.done and "writer" in st.session_state.results:
    st.markdown("---")
    res_col_main, res_col_side = st.columns([6.5, 3.5])

    with res_col_main:
        st.markdown("## 📑 Research Report")
        report_content = st.session_state.results.get("writer", "")
        st.markdown(report_content)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Report (.md)",
            data=report_content,
            file_name=f"research_report_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with res_col_side:
        st.markdown("## 🎯 Critic Review")
        critique_content = st.session_state.results.get("critic", "")
        st.markdown(
            f"""
            <div class="critic-box">
                {critique_content.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🗂 Raw Pipeline Artifacts")

    with st.expander("🔍 Stage 1: Search Agent Raw Findings", expanded=False):
        st.text(st.session_state.results.get("search", "No search output."))

    with st.expander("📄 Stage 2: Reader Agent Scraped Content", expanded=False):
        st.text(st.session_state.results.get("reader", "No scraped output."))


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        ResearchMind Multi-Agent System · Groq Cloud API · LangGraph & LangChain 1.x · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)


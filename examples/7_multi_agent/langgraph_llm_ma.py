import os
from scholarly import scholarly
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# set the OPENAI_API_KEY environment variable
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-key-here"



def search_agent(state):
    query = state.get("query", "applications of AI in education")
    search_results = scholarly.search_pubs(query)
    papers = []
    for i in range(5):  # Limit to 5 results for simplicity
        try:
            paper = next(search_results)
            papers.append({
                "title": paper['bib']['title'],
                "authors": paper['bib']['author'],
                "pub_year": paper['bib'].get('pub_year', 'Unknown'),
                "venue": paper['bib'].get('venue', 'Unknown'),
                "abstract": paper['bib'].get('abstract', 'Unknown')
            })
        except StopIteration:
            break
    state["papers"] = papers
    return state


def filter_agent(state):
    papers = state.get("papers", [])
    if not papers:
        state["filtered_papers"] = []
        return state
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    filtered_papers = []
    
    for paper in papers:
        title = paper['title']
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful academic assistant."),
            ("user", f"Is this paper titled '{title}' relevant to the query '{state['query']}'?. Here is the abstract:\n{paper['abstract']}.\n Reply with 'Yes' or 'No'.")
        ])
        response = llm(prompt.format_messages())
        if "Yes" in response.content:
            filtered_papers.append(paper)
    
    state["filtered_papers"] = filtered_papers
    return state


def query_refinement_agent(state):
    query = state.get("query", "")
    feedback = state.get("feedback", "No relevant papers found.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant."),
        ("user", f"Refine this query: '{query}' to improve search results. Feedback: {feedback}")
    ])
    response = llm(prompt.format_messages())
    state["query"] = response.content.strip()
    return state


def supervisor_agent(state):
    filtered_papers = state.get("filtered_papers", [])
    if len(filtered_papers) <= 1:  # trigger refinement if 0 or 1 paper is found
        state["decision"] = "refine"
        state["feedback"] = f"Only {len(filtered_papers)} relevant paper(s) found. Refining the query."
    else:
        summary = f"Found {len(filtered_papers)} relevant papers:\n"
        for idx, paper in enumerate(filtered_papers, 1):
            summary += f"{idx}. {paper['title']} by {', '.join(paper['authors'])}"
            if paper['pub_year'] and paper['pub_year'] != "NA":
                summary += f" ({paper['pub_year']})"
            if paper['venue'] and paper['venue'] != "NA":
                summary += f" in {paper['venue']}"
            summary += "\n"
        state["decision"] = "finalize"
        state["summary"] = summary
    
    return state


# define the state structure
class State(dict):
    query: str
    papers: list
    filtered_papers: list
    decision: str
    feedback: str
    summary: str

# construct the graph
graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node("SearchAgent", search_agent)
graph_builder.add_node("FilterAgent", filter_agent)
graph_builder.add_node("SupervisorAgent", supervisor_agent)
graph_builder.add_node("QueryRefinementAgent", query_refinement_agent)

# add edges
graph_builder.add_edge(START, "SearchAgent")
graph_builder.add_edge("SearchAgent", "FilterAgent")
graph_builder.add_edge("FilterAgent", "SupervisorAgent")

# add conditional edges from SupervisorAgent
def supervisor_routing(state: State) -> str:
    """Determine the next node after SupervisorAgent."""
    if state.get("decision") == "refine":
        return "QueryRefinementAgent"
    elif state.get("decision") == "finalize":
        return END
    else:
        raise ValueError(f"Unexpected decision: {state.get('decision')}")

graph_builder.add_conditional_edges(
    "SupervisorAgent",
    supervisor_routing
)

graph_builder.add_edge("QueryRefinementAgent", "SearchAgent")

# compile the graph
graph = graph_builder.compile()

# initial query
initial_input = {"query": "service composition roman model"}

# invoke the workflow
final_state = graph.invoke(
    initial_input,
    config={"configurable": {"thread_id": 42}}  # config is optional and can include metadata
)

# Access the final result
if final_state.get("decision") == "finalize":
    print(final_state.get("summary"))
else:
    print("Refined Query Suggested:")
    print(final_state.get("query"))

from langchain_core.runnables.graph import w
graph_png = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
with open("my_graph.png", "wb") as f:
    f.write(graph_png)

from langgraph.graph import END, START, StateGraph

from src.nodes_edges import load_documents, split_documents, map_chunks, identify_pii_items, reprompt_to_identify_more_pii_items, organize_pii_by_file, combine_file_pii_items, postprocess_pii_items
from src.states import InputState, OverallState, OutputState

# Define the graph
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# Add nodes with async functions
builder.add_node("load_documents", load_documents)
builder.add_node("split_documents", split_documents)
builder.add_node("identify_pii_items", identify_pii_items)
builder.add_node("organize_pii_by_file", organize_pii_by_file)
builder.add_node("combine_file_pii_items", combine_file_pii_items)
builder.add_node("postprocess_pii_items", postprocess_pii_items)

# Add edges (with or without conditional routing)
builder.add_edge(START, "load_documents")
builder.add_edge("load_documents", "split_documents")
builder.add_conditional_edges("split_documents", map_chunks, ["identify_pii_items"])
builder.add_conditional_edges("identify_pii_items", reprompt_to_identify_more_pii_items, ["split_documents", "organize_pii_by_file"])
builder.add_edge("organize_pii_by_file", "combine_file_pii_items")
builder.add_edge("combine_file_pii_items", "postprocess_pii_items")
builder.add_edge("postprocess_pii_items", END)

# Compile the graph
graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "PiiDetectionLangGraph"

# Create main agent instance
app = graph
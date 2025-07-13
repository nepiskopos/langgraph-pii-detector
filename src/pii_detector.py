from langgraph.graph import END, START, StateGraph

from src.nodes_edges import _collect_masked_chunks, _combine_file_pii_items, _group_pii_by_file, _identify_pii_items, _load_document, _map_chunks, _map_documents_to_split, _map_input, _mask_documents, _mask_text, _postprocess_pii_items, _route_masked_documents, _should_reprompt, _split_document
from src.states import InputState, OverallState, OutputState

# Define the graph
builder = StateGraph(OverallState, input=InputState, output=OutputState)

# Add nodes with async functions
builder.add_node("load_document", _load_document)
builder.add_node("split_document", _split_document)
builder.add_node("identify_pii_items", _identify_pii_items)
builder.add_node("group_pii_by_file", _group_pii_by_file)
builder.add_node("mask_documents", _mask_documents)
builder.add_node("mask_text", _mask_text)
builder.add_node("collect_masked_chunks", _collect_masked_chunks)
builder.add_node("combine_file_pii_items", _combine_file_pii_items)
builder.add_node("postprocess_pii_items", _postprocess_pii_items)

# Add edges (with or without conditional routing)
builder.add_conditional_edges(START, _map_input, ["load_document"])
builder.add_conditional_edges("load_document", _map_documents_to_split, "split_document")
builder.add_conditional_edges("split_document", _map_chunks, ["identify_pii_items"])
builder.add_edge("identify_pii_items", "group_pii_by_file")
builder.add_conditional_edges("group_pii_by_file", _should_reprompt, ["mask_documents", "combine_file_pii_items"])
builder.add_conditional_edges("mask_documents", _route_masked_documents, ["mask_text", "collect_masked_chunks"])
builder.add_edge("mask_text", "collect_masked_chunks")
builder.add_conditional_edges("collect_masked_chunks", _map_chunks, ["identify_pii_items"])
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
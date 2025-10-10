"""AgentOps DAB Example Project"""
__version__ = "0.1.0"

from .agent_development import (
    create_tool_calling_agent,
    LangGraphChatAgent,
    execute_python_code,
    retrieve_function,
    ask_ai_function,
    summarization_function,
    translate_function,
    get_reference_documentation,
)

from .data_preparation.vector_search.vector_search_utils import (
    vs_endpoint_exists,
    wait_for_vs_endpoint_to_be_ready,
    index_exists,
    wait_for_index_to_be_ready,
)

from .agent_deployment.model_serving.serving import (
    wait_for_model_serving_endpoint_to_be_ready,
)

__all__ = [
    "create_tool_calling_agent",
    "LangGraphChatAgent",
    "execute_python_code",
    "retrieve_function",
    "vs_endpoint_exists",
    "wait_for_vs_endpoint_to_be_ready", 
    "index_exists",
    "wait_for_index_to_be_ready",
    "wait_for_model_serving_endpoint_to_be_ready",
    "__version__",
]

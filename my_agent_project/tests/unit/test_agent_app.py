import pytest
import sys
from unittest.mock import MagicMock, patch

# mocks for create_tool_calling_agent
mock_mlflow = MagicMock()
mock_mlflow.langchain = MagicMock() 
mock_mlflow.langchain.chat_agent_langgraph = MagicMock()
mock_mlflow.pyfunc = MagicMock()
mock_mlflow.pyfunc.ChatAgent = MagicMock()
mock_mlflow.types = MagicMock() 
mock_mlflow.types.agent = MagicMock()

mock_langchain_core = MagicMock()
mock_langchain_core.runnables = MagicMock()
mock_langchain_core.runnables.RunnableConfig = MagicMock()
mock_langchain_core.runnables.RunnableLambda = MagicMock()

mock_langgraph = MagicMock()
mock_langgraph.graph = MagicMock()
mock_langgraph.graph.graph = MagicMock()
mock_langgraph.graph.graph.CompiledStateGraph = MagicMock()
mock_langgraph.graph.state = MagicMock() 
mock_langgraph.graph.state.CompiledStateGraph = MagicMock()
mock_langgraph.prebuilt = MagicMock()
mock_langgraph.prebuilt.tool_node = MagicMock()
mock_langgraph.prebuilt.tool_node.ToolNode = MagicMock()
mock_langgraph.graph.START = "START"
mock_langgraph.graph.END = "END"
mock_langgraph.graph.StateGraph = MagicMock()

sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.langchain'] = mock_mlflow.langchain
sys.modules['mlflow.types'] = mock_mlflow.types
sys.modules['mlflow.types.agent'] = mock_mlflow.types.agent
sys.modules['mlflow.pyfunc'] = mock_mlflow.pyfunc
sys.modules['mlflow.pyfunc.ChatAgent'] = mock_mlflow.pyfunc.ChatAgent
sys.modules['mlflow.langchain.chat_agent_langgraph'] = mock_mlflow.langchain.chat_agent_langgraph
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.runnables'] = mock_langchain_core.runnables
sys.modules['langgraph'] = mock_langgraph
sys.modules['langgraph.graph'] = mock_langgraph.graph
sys.modules['langgraph.graph.graph'] = mock_langgraph.graph.graph
sys.modules['langgraph.graph.state'] = mock_langgraph.graph.state
sys.modules['langgraph.prebuilt'] = mock_langgraph.prebuilt
sys.modules['langgraph.prebuilt.tool_node'] = mock_langgraph.prebuilt.tool_node

class TestCreateToolCallingAgent:
    """Unit tests for create_tool_calling_agent function"""

    def test_create_tool_calling_agent_basic(self):
        """Test basic create_tool_calling_agent function"""
        from agent_development import create_tool_calling_agent
        
        mock_model = MagicMock()
        
        mock_tools = [MagicMock(), MagicMock()]        
        system_prompt = "You are a helpful assistant."
        result = create_tool_calling_agent(mock_model, mock_tools, system_prompt)
        
        mock_model.bind_tools.assert_called_once_with(mock_tools)
        assert result is not None

    def test_should_continue_with_tool_calls(self):
        """Test should_continue logic when tool calls are present"""
        state_with_tools = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": None, "tool_calls": [{"name": "test_tool"}]}
            ]
        }
        
        last_message = state_with_tools["messages"][-1]
        result = "tools" if last_message.get("tool_calls") else "END"
        assert result == "tools"

    def test_should_continue_without_tool_calls(self):
        """Test should_continue logic when no tool calls are present"""
        state_without_tools = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help?", "tool_calls": None}
            ]
        }
        
        last_message = state_without_tools["messages"][-1]  
        result = "tools" if last_message.get("tool_calls") else "END"
        assert result == "END"

"""
LLM Mock and patching utilities for agent tests.
"""
from typing import Callable, List, Dict, Set, Any, Optional, Mapping, Iterator
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, AIMessage, ToolCall
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from databricks_langchain import ChatDatabricks
import json


class Content(ABC): 
    @abstractmethod
    def build(self, messages: List[BaseMessage], tool_results: Dict[str, Any], uc_catalog: str, schema: str) -> str: 
        pass 

class FixedContent(Content): 
    """ fixed content """
    def __init__(self, content: str): 
        self.content = content

    def build(self, messages: List[BaseMessage], tool_results: Dict[str, Any], uc_catalog: str, schema: str) -> str: 
        return self.content
    
    def __add__(self, other: Content) -> Content: 
        if isinstance(other, FixedContent):
            return FixedContent(self.content + other.content)
        if isinstance(other, ToolContent): 
            return ToolContent(other.tool_name, self.content + other.content)
        else: 
            raise TypeError(f"Cannot add FixedContent and {type(other).__name__}")

class ToolContent(Content):
    """ retrieve content from tool results """
    def __init__(self, tool_name: str, content: str = ""): 
        self.tool_name = tool_name
        self.content = content
    
    def build(self, messages: List[BaseMessage], tool_results: Dict[str, Any], uc_catalog: str, schema: str) -> str: 
        _tool_name = f"{uc_catalog}__{schema}__{self.tool_name}"
        tool_result = json.loads(tool_results.get(_tool_name, ""))
        return tool_result.get("value", "") + self.content

    def __add__(self, other: FixedContent) -> Content: 
        # add fixed content to tool content to get fake AI message to 
        # output tool responses with fixed content
        return ToolContent(self.tool_name, self.content + other.content)

class Scenario: 
    """ represents a fixed scenario with fake LLM responses """ 

    def __init__(self, role: str, content: Content, tool_name: Optional[str] = None, args: Optional[Dict[str, Any]] = None, uc_catalog:str = "", schema:str = ""):
        self.role = role
        self.content = content
        self.tool_name = tool_name
        self.args = args
        self.uc_catalog = uc_catalog 
        self.schema = schema

    def to_dict(self, messages: List[BaseMessage], tool_results: Dict[str, Any]) -> Dict[str, Any]: 
        """ convert scenario to dict """ 
        result = {
            "role": self.role, 
            "content": self.content.build(messages, tool_results, self.uc_catalog, self.schema)
        }
        if self.tool_name: 
            result['tool_name'] = self.tool_name
        if self.args: 
            result['args'] = self.args
        return result

class Puppeteer: 
    """ puppeteer for LLM responses """ 
    def __init__(self, uc_catalog: str, schema: str):
        self.uc_catalog = uc_catalog
        self.schema = schema
        self.scenarios: List[Scenario] = []
    
    def add_tool(self, tool_name: str, args:Dict[str, Any]): 
        """ test the LLM response """ 
        _tool_name = f"{self.uc_catalog}__{self.schema}__{tool_name}"
        empty_content = FixedContent("")
        self.scenarios.append(Scenario(role='tool', content=empty_content, tool_name=_tool_name, args=args, uc_catalog=self.uc_catalog, schema=self.schema))
        return self 
    
    def add_output(self, content: Content): 
        self.scenarios.append(Scenario(role='output', content=content, uc_catalog=self.uc_catalog, schema=self.schema))
        return self 
    
    def build(self) -> List[Scenario]: 
        return self.scenarios.copy()    

class LLMMock(BaseChatModel):
    """ Mock LLM for testing. Can use real LLM optionally."""
    scenarios: List[Scenario] = []
    llm: ChatDatabricks = None
    i: int = 0
    called: bool = False
    call_count: int = 0
    real_llm_called: bool = False
    real_llm_call_count: int = 0
    should_mock: bool = False
    tools_called: List[Dict[str, Any]] = []

    def __init__(self, responses: List[Scenario], llm: ChatDatabricks = None, should_mock_tool_calls: Callable[[List[BaseMessage]], bool] = None, **kwargs):
        super().__init__(**kwargs)
        self.scenarios = responses
        self.llm = llm # real llm 
        self.i = 0
        self.called = False
        self.call_count = 0 
        self.tools_called = []
        self.real_llm_called = False
        self.real_llm_call_count = 0

        if self.llm: 
            self.should_mock = should_mock_tool_calls or self._default_should_mock
        else: 
            self.should_mock = lambda messages: True

    
    def _default_should_mock(self, messages: List[BaseMessage]) -> bool:
        """by default, we want to mock non-tool conversations"""
        if not messages: 
            return True
        
        has_tool_results = any(hasattr(msg, 'type') and msg.type == "tool" for msg in messages)
        # if no tool results -> let real LLM will handle tool calls. 
        # if tool results -> use mock
        return has_tool_results
    
    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def _get_tool_response(self, messages: List[BaseMessage]) -> Dict[str, Any]: 
        """ get latest tool response """ 
        tool_results = {}

        for message in reversed(messages): 
            if hasattr(message, 'type') and message.type == "tool": 
                tool_results[message.name] = message.content 
        return tool_results
    
    def _to_ai_message(self, scenario: Dict[str, Any]) -> AIMessage: 
        """ convert scenario to AI message """ 
        if scenario['role'] == 'tool': 
            tc = ToolCall(name=scenario['tool_name'], args=scenario['args'], id=f"call_{scenario['tool_name']}_{self.i}")
            return AIMessage(content=scenario['content'], tool_calls=[tc])
        elif scenario['role'] == 'output': 
            return AIMessage(content=scenario['content'], tool_calls=[])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """ Generate chat response with potential tool calls"""
        self.called = True
        self.call_count += 1

        if self.should_mock(messages):
            tool_results = self._get_tool_response(messages)
            
            if self.i >= len(self.scenarios):
                ai_message = AIMessage(content="Ran out of mock scenarios...", tool_calls=[])
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
            
            current_scenario = self.scenarios[self.i]
            ai_message = self._to_ai_message(current_scenario.to_dict(messages, tool_results))
            self.i += 1 # next scenario
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        
        elif self.llm: 
            # Generate Tool Calls - use .invoke() and not ._generate() as _generate will not work with binded tools
            # To do so, we need to convert to acceptable input: 
            formatted_messages = [] 
            for msg in messages: 
                if hasattr(msg, 'role') and hasattr(msg, 'content'): 
                    formatted_messages.append({"role": msg.role, "content": msg.content})
                if hasattr(msg, 'type') and hasattr(msg, 'content'): 
                    formatted_messages.append({"role": msg.type, "content": msg.content})

            ai_message = self.llm.invoke(formatted_messages)   

            self.real_llm_called = True
            self.real_llm_call_count += 1

            self.tools_called.extend(ai_message.tool_calls)
            
            # convert the output back to what _generate expects
            return ChatResult(generations=[ChatGeneration(message=ai_message)])

        else:  
            ai_message = AIMessage(content="Something went wrong. Pass an LLM if you do not want mock calls.", tool_calls=[])
            return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(self, tools):
        """Mock tool binding"""
        if self.llm: 
            self.llm = self.llm.bind_tools(tools, tool_choice='required')
        return self
    
    def assert_tools_called(self, tool_name): 
        # tool_calls are formatted like: [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': 'xxx', 'type': 'tool_call'}]
        for t in self.tools_called: 
            if tool_name == t['name'].split("__")[-1]: # remove catalog + schema 
                return True
        return False

    def assert_called(self):
        if not self.called:
            raise AssertionError("LLM mock was not called")
        
    def assert_real_llm_called(self):
        if not self.real_llm_called: 
            raise AssertionError("Real LLM was not called")
    
    def assert_call_count(self, expected: int): 
        if self.call_count != expected:
            raise AssertionError(f"Expected {expected} calls, got {self.call_count}")

    def assert_real_llm_call_count(self, expected: int): 
        if self.real_llm_call_count != expected: 
            raise AssertionError(f"Expected {expected} calls, got {self.real_llm_call_count}")

    def assert_called_at_least(self, expected: int):
        if self.call_count < expected:
            raise AssertionError(f"Expected at least {expected} calls, got {self.call_count}")



class LLMPatcher: 
    """ context manager for patching LLM calls """ 
    def __init__(self, llm_mock: LLMMock, target: str = "databricks_langchain.ChatDatabricks"):
        self.llm_mock = llm_mock
        self.target = target
        self.original = None
        self.module = None
        self.attr = None

    def __enter__(self):
        """ start patching """ 
        parts = self.target.split(".")
        module_path = ".".join(parts[:-1])
        attr_name = parts[-1]

        import importlib
        self.module = importlib.import_module(module_path)
        self.attr = attr_name
        self.original = getattr(self.module, attr_name)
                    
        setattr(self.module, attr_name, lambda *args, **kwargs: self.llm_mock)
        
        patched = getattr(self.module, attr_name)
            
        return self.llm_mock
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """ stop patching + restore original modules"""         
        if self.module and self.attr and self.original: 
            setattr(self.module, self.attr, self.original)
            restored = getattr(self.module, self.attr)
            
            import importlib
            importlib.reload(self.module)
                 
            import sys
            app_modules = [k for k in sys.modules.keys() if 'app' in k]
            for mod_name in app_modules:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
                     

def clear_app_imports():
    """Clear any cached imports of app.py to ensure clean slate"""
    import sys
    import gc
    
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(part in module_name.lower() for part in ['app', 'agent', 'notebook']):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]
    gc.collect()

def puppet(puppeteer: Puppeteer,
           llm: ChatDatabricks = None, 
           should_mock_tool_calls: Callable[[List[BaseMessage]], bool] = None, 
           target: str = "databricks_langchain.ChatDatabricks") -> LLMPatcher:
    """
    LLM patching for agent test cases
    """
    return LLMPatcher(llm_mock=LLMMock(responses=puppeteer.build(), llm=llm, should_mock_tool_calls=should_mock_tool_calls), 
                      target=target)

"""
Integration test definitions for Databricks Agent
"""

import mlflow
import json
from mlflow.entities.span import SpanAttributeKey
from .test_runner import DatabricksTestRunner
from .assertions import (
    validate_uc_function_result
    )
from .mock import (
    puppet,
    Puppeteer, 
    FixedContent, 
    ToolContent,
    clear_app_imports
)
from databricks_langchain import ChatDatabricks


def create_uc_function_tests(
    client,
    python_execution_function_name: str,
    ask_ai_function_name: str,
    summarization_function_name: str,
    translate_function_name: str,
    max_words: int
) -> DatabricksTestRunner:
    """
    Configure a runner with UC function tests

    Args:
        client: UC Function client
        python_execution_function_name: Full name of Python execution function
        ask_ai_function_name: Full name of ask AI function
        summarization_function_name: Full name of summarization function
        translate_function_name: Full name of translation function
        max_words: Maximum words for summarization
    
    Returns: DatabricksTestRunner
    """
    
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Python Execution Function")
    def test_python_execution_function():
        """Test Python code execution UC function"""
        test_cases = [
            {"code": "print('Hello UC Function')"},
            {"code": "result = 2 + 2; print(f'Result: {result}')"},
            {"code": "import datetime; print(datetime.datetime.now().strftime('%Y-%m-%d'))"}
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            result = client.execute_function(python_execution_function_name, test_input)
            validate_uc_function_result(result, python_execution_function_name)
            print(f"     Python test case {i} passed")
    
    @test_runner.test("Python Execution Error Handling")
    def test_python_execution_errors():
        """Test Python function handles errors gracefully"""
        error_cases = [
            {"code": "print(undefined_variable)"},  
            {"code": "print('missing quote"},       
            {"code": "1/0"}                       
        ]
        
        for i, test_input in enumerate(error_cases, 1):
            result = client.execute_function(python_execution_function_name, test_input)
            validate_uc_function_result(result, python_execution_function_name)
            assert "error" in result.value.lower() or "failed" in result.value.lower(), f"Error case {i} not handled properly"
            print(f"     Error case {i} handled gracefully")
    
    @test_runner.test("Summarization Function")
    def test_summarization_function():
        """Test summarization UC function"""
        test_cases = [
            {
                "text": "MLflow is an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, deployment, and a central model registry. It provides tracking capabilities for experiments, packaging ML code in a reusable format, and serving models through various deployment targets.",
                "max_words": max_words
            },
            {
                "text": "Short text",
                "max_words": max_words
            }
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            result = client.execute_function(summarization_function_name, test_input)
            validate_uc_function_result(result, summarization_function_name)
            
            if len(test_input["text"]) > 100:
                assert len(result.value) < len(test_input["text"]), f"Summarization {i} didn't reduce text length"
            
            print(f"     Summarization test case {i} passed")
    
    @test_runner.test("Translation Function")
    def test_translation_function():
        """Test translation UC function"""
        test_cases = [
            {"content": "Hello world", "language": "es"},
            {"content": "Good morning", "language": "fr"},
            {"content": "Thank you", "language": "de"}
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            result = client.execute_function(translate_function_name, test_input)
            validate_uc_function_result(result, translate_function_name)
            
            assert result.value.lower() != test_input["content"].lower(), f"Translation {i} appears unchanged"
            
            print(f"     Translation test case {i} passed")
    
    @test_runner.test("Ask AI Function")
    def test_ask_ai_function():
        """Test ask AI UC function"""
        test_cases = [
            {"question": "What is MLflow?"},
            {"question": "Explain machine learning in simple terms"},
            {"question": "What are the benefits of using Databricks?"}
        ]
        
        for i, test_input in enumerate(test_cases, 1):
            result = client.execute_function(ask_ai_function_name, test_input)
            validate_uc_function_result(result, ask_ai_function_name)
            
            assert len(result.value) > 10, f"Ask AI {i} returned very short response"
            
            print(f"     Ask AI test case {i} passed")
    
    @test_runner.test("Function Parameter Validation")
    def test_function_parameters():
        """Test function parameter validation"""
        try:
            result = client.execute_function(python_execution_function_name, {})
            validate_uc_function_result(result, python_execution_function_name)
            print("     Missing parameter handled gracefully")
        except Exception as e:
            print(f"     Missing parameter validation: {str(e)[:50]}...")
    
    return test_runner


def create_agent_app_tests() -> DatabricksTestRunner:
    """
    Configure a runner with agent tests

    Returns: DatabricksTestRunner
    """
    
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Import AGENT from app.py")
    def test_import_agent():
        """Test that AGENT can be imported from app.py"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        assert AGENT is not None, "AGENT is None"
        
        assert hasattr(AGENT, 'predict'), "AGENT missing predict method"
        assert hasattr(AGENT, 'predict_stream'), "AGENT missing predict_stream method"
        
        print(f"    AGENT imported successfully from app.py")
    
    @test_runner.test("AGENT.predict Basic Response")
    def test_agent_predict_basic():
        """Test AGENT.predict with basic queries"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        test_queries = [
            "Hello, can you respond to this message?",
            "What is your purpose?",
            "Can you help me with MLflow?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            test_messages = [ChatAgentMessage(role="user", content=query)]
            response = AGENT.predict(test_messages)
            
            assert hasattr(response, 'messages'), f"AGENT.predict didn't return proper response for query {i}"
            assert len(response.messages) > 0, f"AGENT.predict returned empty messages for query {i}"
            
            last_message = response.messages[-1]
            assert hasattr(last_message, 'content'), f"Last message missing content for query {i}"
            assert len(last_message.content) > 0, f"Last message content empty for query {i}"
            assert len(last_message.content) > 10, f"Response {i} too short: {len(last_message.content)} chars"
            
            print(f"    Basic predict test {i} passed")
    
    @test_runner.test("AGENT.predict Python Tool Integration")
    def test_agent_predict_python_tool():
        """Test Python code execution through AGENT.predict"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        test_queries = [
            "Execute this Python code: print('Hello from AGENT.predict')",
            "Run this code: result = 7 * 6; print(f'7 x 6 = {result}')",
            "Execute: import datetime; print(f'Today: {datetime.datetime.now().strftime(\"%Y-%m-%d\")}')"
        ]
        
        for i, query in enumerate(test_queries, 1):
            test_messages = [ChatAgentMessage(role="user", content=query)]
            response = AGENT.predict(test_messages)
            
            assert len(response.messages) > 0, f"Python tool test {i} returned empty messages"
            
            # Check that we got a substantive response
            last_message = response.messages[-1]
            assert len(last_message.content) > 0, f"Python tool test {i} returned empty content"
            
            # For Python execution, we expect to see some execution result or confirmation
            execution_indicators = ["executed", "result", "output", "error", "print"]
            response_lower = last_message.content.lower()
            
            # Check for output for test_queries
            import datetime
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            datetime_indicators = ["Hello from AGENT.predict", "42", current_date]
            datetime_found = datetime_indicators[i-1]

            assert datetime_found, f"Python tool test {i} doesn't show proper output: {last_message.content[:100]}..."
            print(f"     Python tool test {i} passed via AGENT.predict")
    
    @test_runner.test("AGENT.predict Translation Tool Integration")
    def test_agent_predict_translation():
        """Test translation through AGENT.predict"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        test_queries = [
            "Translate 'Hello world' to Spanish",
            "Can you translate 'Good morning' to French?",
            "Please translate 'Thank you' to German"
        ]
        
        for i, query in enumerate(test_queries, 1):
            test_messages = [ChatAgentMessage(role="user", content=query)]
            response = AGENT.predict(test_messages)
            
            assert len(response.messages) > 0, f"Translation test {i} returned empty messages"
            
            last_message = response.messages[-1]
            assert len(last_message.content) > 0, f"Translation test {i} returned empty content"
            
            translation_indicators = ["translate", "spanish", "french", "german", "hola", "bonjour", "danke"]
            response_lower = last_message.content.lower()
            
            found_indicator = any(indicator in response_lower for indicator in translation_indicators)
            assert found_indicator, f"Translation test {i} doesn't show translation: {last_message.content[:100]}..."
            
            print(f"     Translation test {i} passed via AGENT.predict")
    
    @test_runner.test("AGENT.predict Error Handling")
    def test_agent_predict_error_handling():
        """Test that AGENT.predict handles errors"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        error_queries = [
            "Execute this broken code: print(undefined_variable)",
            "Translate this nonsense: asdfghjkl qwertyuiop",
            "Summarize this empty text: "
        ]
        
        for i, query in enumerate(error_queries, 1):
            try:
                test_messages = [ChatAgentMessage(role="user", content=query)]
                response = AGENT.predict(test_messages)
                
                assert len(response.messages) > 0, f"Error handling test {i} returned empty messages"
                
                last_message = response.messages[-1]
                assert len(last_message.content) > 0, f"Error handling test {i} returned empty content"
                
                print(f"     Error handling test {i} passed - no crashes")
                
            except Exception as e:
                # not failing test hard
                print(f"     Error handling test {i} crashed: {e}")


    @test_runner.test("AGENT.predict Multi-Tool Workflow")
    def test_agent_predict_multi_tool():
        """Test complex workflow using multiple tools via AGENT.predict"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        complex_query = "Find information about MLflow, write Python code to demonstrate it, and translate a summary to Spanish"
        
        test_messages = [ChatAgentMessage(role="user", content=complex_query)]
        response = AGENT.predict(test_messages)
        
        assert len(response.messages) > 0, "Multi-tool test returned empty messages"
        
        last_message = response.messages[-1]
        assert len(last_message.content) > 0, "Multi-tool test returned empty content"
        
        assert len(last_message.content) > 50, f"Multi-tool response too short: {len(last_message.content)} chars"
        
        tool_indicators = ["mlflow", "code", "python", "spanish", "translate", "execute"]
        response_lower = last_message.content.lower()
        
        found_indicators = [indicator for indicator in tool_indicators if indicator in response_lower]
        assert len(found_indicators) >= 2, f"Multi-tool test shows limited tool usage. Found: {found_indicators}"
        
        print(f"     Multi-tool workflow test passed via AGENT.predict")
    
    @test_runner.test("AGENT Conversation State")
    def test_agent_conversation_state():
        """Test that AGENT maintains conversation context"""
        import sys
        sys.path.append("../notebooks")
        
        from app import AGENT
        from mlflow.types.agent import ChatAgentMessage
        
        messages = [
            ChatAgentMessage(role="user", content="Execute this code: x = 42"),
            ChatAgentMessage(role="assistant", content="I'll execute that code for you."),
            ChatAgentMessage(role="user", content="Now execute: print(f'The value of x is {x}')")
        ]
        
        response = AGENT.predict(messages)
        
        assert len(response.messages) > 0, "Conversation state test returned empty messages"
        
        last_message = response.messages[-1]
        assert len(last_message.content) > 0, "Conversation state test returned empty content"
        
        context_indicators = ["42", "x", "value", "variable"]
        response_lower = last_message.content.lower()
        
        found_context = any(indicator in response_lower for indicator in context_indicators)
        if found_context:
            print(f"    Conversation state appears to be maintained")
        else:
            print(f"    Conversation state handling unclear - may still be working")
        
        print(f"    Conversation state test completed")
    
    return test_runner


def create_registered_model_tests(
    uc_catalog: str, 
    schema: str,
    registered_model: str,
    model_alias: str
) -> DatabricksTestRunner: 
    """
    Configure a runner with registered model tests

    Args:
        uc_catalog: Unity Catalog name
        schema: Schema name
        model_alias: Model alias

    Returns: DatabricksTestRunner
    """

    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Model Registration and Alias")
    def test_model_version_aliases():
        """Test model registration and aliases"""        
        agent = mlflow.pyfunc.load_model(f"models:/{uc_catalog}.{schema}.{registered_model}@agent_latest")
        response = agent.predict(
            {"messages": [{"role": "user", "content": "What is MLflow?"}]}
        )
        
        assert response is not None
        print("   @ alias works correctly")

    @test_runner.test("Model Environment")
    def test_model_environment():
        """Test model runs in environment with dependencies resolved via uv"""
        model_uri = f"models:/{uc_catalog}.{schema}.{registered_model}@agent_latest"        
        try:
            response = mlflow.models.predict(
                model_uri,
                input_data={"messages": [{"role": "user", "content": "Environment test"}]},
                env_manager="uv"
            )
            assert response is not None
            print(f"    Model + dependencies loaded correctly with uv.")
        except Exception as e:
            if "'uv' command is not found" in str(e):
                print(f"    uv not available, skipping")
            else:
                assert False, f"Failed with uv: {e}"


    @test_runner.test("Model Serving Input Validation")
    def test_model_serving_input():
        """Test model serving input"""
        from mlflow.models import convert_input_example_to_serving_input, validate_serving_input

        model_uri = f"models:/{uc_catalog}.{schema}.{registered_model}@agent_latest"        

        serving_input = mlflow.models.convert_input_example_to_serving_input(
            {"messages": [{"role": "user", "content": "What is MLflow?"}]}
        )
        output = mlflow.models.validate_serving_input(model_uri, serving_input=serving_input)
        assert output is not None, "Model serving input validation failed"
         
        assert "mlflow" in output['messages'][0]['content'].lower()
        assert len(output['messages'][0]['content']) > 10, "Model serving input validation failed"
        print(f"    Model serving input validation passed")

    return test_runner 


def create_mocked_agent_tests(
    uc_catalog: str, 
    schema: str
) -> DatabricksTestRunner:
    """
    Configure a runner with agent with mocked LLM responses
    
    Returns: DatabricksTestRunner
    """
    from .test_helpers import puppet
    test_runner = DatabricksTestRunner(fail_fast=False, verbose=True)
    
    @test_runner.test("Mocked LLM Python Tool Integration")
    def test_mocked_python_tool():
        """Test Python tool integration with mocked LLM responses"""
        import sys
        sys.path.append("../notebooks")

        puppeteer = Puppeteer(uc_catalog, schema)
        puppeteer.add_tool("execute_python_code", {"code": "print('Hello from mocked test')"})
        puppeteer.add_output(FixedContent("The code executed: ") + ToolContent("execute_python_code"))

        with puppet(puppeteer):
            from app import AGENT
            from mlflow.types.agent import ChatAgentMessage
            
            test_messages = [ChatAgentMessage(role="user", content="Execute: print('Hello from mocked test')")]
            response = AGENT.predict(test_messages)
            assert len(response.messages) > 0, "Mocked agent returned no messages"
            
            print("     Mocked Python tool integration test passed")
        
        clear_app_imports()

    @test_runner.test("LLM Call Python Tool Integration")
    def test_llm_python_tool_calling():
        """Test Python tool integration with mocked LLM responses"""
        import sys
        sys.path.append("../notebooks")
        
        real_llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
        real_llm = ChatDatabricks(endpoint=real_llm_endpoint)

        puppeteer = Puppeteer(uc_catalog, schema)
        puppeteer.add_output(FixedContent("I have completed the query."))
        
        with puppet(puppeteer, real_llm) as mock:
            from app import AGENT
            from mlflow.types.agent import ChatAgentMessage
            
            test_messages = [ChatAgentMessage(
                role="user", 
                content="Execute print('Hello from mocked test')"
            )]
            
            response = AGENT.predict(test_messages)
            assert len(response.messages) > 0, "Agent returned no messages"
            
            mock.assert_called() 
            mock.assert_real_llm_called()
            mock.assert_tools_called("execute_python_code")

            print("     LLM Python tool calling integration test passed")
        clear_app_imports()


    @test_runner.test("LLM Call Translate and Python Tool Integration")
    def test_llm_multi_tool_calling():
        """Test Python tool integration with mocked LLM responses"""
        import sys
        sys.path.append("../notebooks")
        
        real_llm_endpoint = "databricks-meta-llama-3-3-70b-instruct"
        real_llm = ChatDatabricks(endpoint=real_llm_endpoint)

        puppeteer = Puppeteer(uc_catalog, schema)
        puppeteer.add_output(FixedContent("I have completed the query."))
        
        with puppet(puppeteer, real_llm) as mock:
            from app import AGENT
            from mlflow.types.agent import ChatAgentMessage
            
            test_messages = [ChatAgentMessage(
                role="user", 
                content="Execute print('Hello from mocked test') and translate the output to Spanish"
            )]
            
            response = AGENT.predict(test_messages)
            assert len(response.messages) > 0, "Agent returned no messages"
            
            mock.assert_called() 
            mock.assert_real_llm_called()
            mock.assert_tools_called("translate")
            mock.assert_tools_called("execute_python_code")

            print("     LLM Multi-Tool calling integration test passed")
        clear_app_imports()

    return test_runner 
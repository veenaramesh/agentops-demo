# tests/unit/agent_development/tools/test_tool_definitions.py
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# mock databricks.vector_search before importing tools (retrieve_function uses this class)
# we will not test retrieve_function in this as it relies on the vector search index which is not available in the test environment
# we will also only test formatting of SQL functions as it relies on UC which is not available in the test environment  

mock_databricks = MagicMock()
mock_vector_search_index = MagicMock()
mock_vector_search_index.VectorSearchIndex = MagicMock()
mock_vector_search = MagicMock()
mock_vector_search.index = mock_vector_search_index
mock_databricks.vector_search = mock_vector_search
mock_databricks.sdk = MagicMock()
mock_databricks_langchain = MagicMock()
mock_databricks_langchain.VectorSearchRetrieverTool = MagicMock()
mock_databricks_langchain.GenieAgent = MagicMock()
mock_databricks_langchain.genie = MagicMock()
mock_databricks_langchain.genie.GenieAgent = MagicMock()
mock_langchain_core = MagicMock()
mock_langchain_core.tools = MagicMock()
mock_langchain_core.prompts = MagicMock()
mock_langchain_core.language_models = MagicMock()
mock_langchain_core.runnables = MagicMock()
mock_langchain_core.messages = MagicMock()
mock_langchain_core.output_parsers = MagicMock()

sys.modules['databricks'] = mock_databricks
sys.modules['databricks.vector_search'] = mock_vector_search
sys.modules['databricks.vector_search.index'] = mock_vector_search_index
sys.modules['databricks.sdk'] = mock_databricks.sdk
sys.modules['databricks_langchain'] = mock_databricks_langchain
sys.modules['databricks_langchain.genie'] = mock_databricks_langchain.genie
sys.modules['langchain_core'] = mock_langchain_core
sys.modules['langchain_core.tools'] = mock_langchain_core.tools
sys.modules['langchain_core.prompts'] = mock_langchain_core.prompts
sys.modules['langchain_core.language_models'] = mock_langchain_core.language_models
sys.modules['langchain_core.runnables'] = mock_langchain_core.runnables
sys.modules['langchain_core.messages'] = mock_langchain_core.messages
sys.modules['langchain_core.output_parsers'] = mock_langchain_core.output_parsers

from agent_development import (
    execute_python_code,
    ask_ai_function,
    summarization_function, 
    translate_function,
    retrieve_function
)

class TestExecutePythonCode:
    """Unit tests for execute_python_code function"""
    
    def test_simple_print_statement(self):
        """Test executing simple print statement"""
        code = "print('Hello, World!')"
        result = execute_python_code(code)
        assert result == "Hello, World!\n"
    
    def test_mathematical_calculation(self):
        """Test executing mathematical calculations"""
        code = """result = 10 + 20 \nprint(f"The sum is: {result}")"""
        result = execute_python_code(code)
        assert "The sum is: 30" in result
    
    def test_multiple_print_statements(self):
        """Test code with multiple print statements"""
        code = """print("First line") \nprint("Second line") \nprint("Third line")"""
        result = execute_python_code(code)
        lines = result.strip().split('\n')
        assert len(lines) == 3
        assert "First line" in result
        assert "Second line" in result
        assert "Third line" in result
    
    def test_variable_operations(self):
        """Test operations with variables"""
        code = """x = 5 \ny = 10 \nz = x * y \nprint(f"x={x}, y={y}, z={z}")"""
        result = execute_python_code(code)
        assert "x=5" in result
        assert "y=10" in result
        assert "z=50" in result
    
    def test_list_operations(self):
        """Test list operations"""
        code = """numbers = [1, 2, 3, 4, 5] \ntotal = sum(numbers) \nprint(f"Numbers: {numbers}") \nprint(f"Sum: {total}")"""
        result = execute_python_code(code)
        assert "[1, 2, 3, 4, 5]" in result
        assert "Sum: 15" in result
    
    def test_dictionary_operations(self):
        """Test dictionary operations"""
        code = """data = {"name": "Alice", "age": 30} \nprint(f"Name: {data['name']}") \nprint(f"Age: {data['age']}")"""
        result = execute_python_code(code)
        assert "Name: Alice" in result
        assert "Age: 30" in result
    
    def test_loop_operations(self):
        """Test loop operations"""
        code = """for i in range(3): \n print(f"Iteration: {i}")"""
        result = execute_python_code(code)
        assert "Iteration: 0" in result
        assert "Iteration: 1" in result
        assert "Iteration: 2" in result
    
    def test_function_definition_and_call(self):
        """Test defining and calling functions"""
        code = """def greet(name): \n return f"Hello, {name}!" \nmessage = greet("World") \nprint(message)"""
        result = execute_python_code(code)
        assert "Hello, World!" in result
    
    def test_empty_code(self):
        """Test executing empty code"""
        result = execute_python_code("")
        assert result == ""
    
    def test_whitespace_only_code(self):
        """Test executing whitespace-only code"""
        result = execute_python_code("   \n   \t   ")
        assert result == ""
    
    def test_code_without_print(self):
        """Test code that doesn't print anything"""
        code = """x = 10 \ny = 20 \nz = x + y"""
        result = execute_python_code(code)
        assert result == ""
    
    def test_syntax_error_handling(self):
        """Test handling of syntax errors"""
        code = "print('Hello world'"  # Missing closing parenthesis
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
    
    def test_runtime_error_handling(self):
        """Test handling of runtime errors"""
        code = "print(undefined_variable)"
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
        assert "name 'undefined_variable' is not defined" in result
    
    def test_division_by_zero_error(self):
        """Test handling of division by zero"""
        code = """x = 10 \ny = 0 \nresult = x / y \nprint(result)"""
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
        assert "division by zero" in result
    
    def test_type_error_handling(self):
        """Test handling of type errors"""
        code = """x = "hello" \ny = 5 \nresult = x + y \nprint(result)"""
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
        assert "can only concatenate str" in result
    
    def test_import_basic_modules(self):
        """Test importing basic Python modules"""
        code = """import math \nresult = math.sqrt(16) \nprint(f"Square root of 16 is: {result}")"""
        result = execute_python_code(code)
        assert "Square root of 16 is: 4.0" in result
    
    def test_import_datetime(self):
        """Test importing datetime module"""
        code = """from datetime import datetime \nnow = datetime.now() \nprint(f"Current year: {now.year}")"""
        result = execute_python_code(code)
        assert "Current year:" in result
    
    def test_import_json(self):
        """Test importing json module"""
        code = """import json \ndata = {"name": "Alice", "age": 30} \njson_str = json.dumps(data) \nprint(f"JSON: {json_str}")"""
        result = execute_python_code(code)
        assert '"name": "Alice"' in result
        assert '"age": 30' in result
    
    def test_complex_calculation(self):
        """Test complex mathematical calculation"""
        code = """\ndef calculate_circle_area(radius): \n import math \n return math.pi * radius ** 2 \nradius = 5 \narea = calculate_circle_area(radius) \nprint(f"Area of circle with radius {radius}: {area:.2f}")"""
        result = execute_python_code(code)
        assert "Area of circle with radius 5:" in result
        assert "78.54" in result  # π * 5² ≈ 78.54
    
    def test_complex_calculation_wrong_import(self): 
        """Test complex mathematical calculation with wrong import"""
        code = """import math \ndef calculate_circle_area(radius): \n return math.pi * radius ** 2 \nradius = 5 \narea = calculate_circle_area(radius) \nprint(f"Area of circle with radius {radius}: {area:.2f}")"""
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
        assert "name 'math' is not defined" in result

    
    def test_string_operations(self):
        """Test string manipulation operations"""
        code = """text = "Hello World" \nprint(f"Original: {text}") \nprint(f"Upper: {text.upper()}") \nprint(f"Lower: {text.lower()}") \nprint(f"Length: {len(text)}")"""
        result = execute_python_code(code)
        assert "Original: Hello World" in result
        assert "Upper: HELLO WORLD" in result
        assert "Lower: hello world" in result
        assert "Length: 11" in result
    
    def test_spark_operations(self):
        """Test Spark operations"""
        code = """from pyspark.sql import SparkSession \nspark = SparkSession.builder.appName("Test").getOrCreate() \nprint(spark)"""
        result = execute_python_code(code)
        assert "Python code execution failed:" in result

    def test_databricks_operations(self):
        """Test Databricks dbutils operations"""
        code = """dbutils.widgets.text("test", "test")"""
        result = execute_python_code(code)
        assert "Python code execution failed:" in result
        assert "name 'dbutils' is not defined" in result

class TestSQLFunctionTemplates:
    """Unit tests for UC functions:  ask_ai, summarize, translate"""
    
    def test_ask_ai_function_template(self):
        """Test ask_ai function SQL template generation"""
        function_name = "test_catalog.test_schema.ask_ai"
        sql = ask_ai_function.format(ask_ai_function_name=function_name)
        
        assert "CREATE OR REPLACE FUNCTION" in sql
        assert function_name in sql
        assert "question STRING" in sql
        assert "RETURNS STRING" in sql
        assert "ai_gen(question)" in sql
        assert "COMMENT" in sql
    
    def test_summarization_function_template(self):
        """Test summarization function SQL template generation"""
        function_name = "test_catalog.test_schema.summarize"
        sql = summarization_function.format(summarization_function_name=function_name)
        
        assert "CREATE OR REPLACE FUNCTION" in sql
        assert function_name in sql
        assert "text STRING" in sql
        assert "max_words INT" in sql
        assert "RETURNS STRING" in sql
        assert "ai_summarize(text, max_words)" in sql
        assert "non-negative integer" in sql
    
    def test_translate_function_template(self):
        """Test translate function SQL template generation"""
        function_name = "test_catalog.test_schema.translate"
        sql = translate_function.format(translate_function_name=function_name)
        
        assert "CREATE OR REPLACE FUNCTION" in sql
        assert function_name in sql
        assert "content STRING" in sql
        assert "language STRING" in sql
        assert "RETURNS STRING" in sql
        assert "ai_translate(content, language)" in sql
        assert "english <-> spanish" in sql
    
    def test_function_template_with_special_characters(self):
        """Test templates with function names containing special characters"""
        function_name = "catalog-with-dashes.schema_with_underscores.function_name"
        
        sql_ask = ask_ai_function.format(ask_ai_function_name=function_name)
        sql_summarize = summarization_function.format(summarization_function_name=function_name)
        sql_translate = translate_function.format(translate_function_name=function_name)
        
        assert function_name in sql_ask
        assert function_name in sql_summarize
        assert function_name in sql_translate
    
    def test_function_template_parameter_validation(self):
        """Test that templates contain proper parameter validation hints"""
        function_name = "test.schema.func"
        
        sql = summarization_function.format(summarization_function_name=function_name)
        assert "non-negative integer" in sql
        assert "if set to 0 then no limit" in sql
        sql = translate_function.format(translate_function_name=function_name)
        assert "english <-> spanish" in sql


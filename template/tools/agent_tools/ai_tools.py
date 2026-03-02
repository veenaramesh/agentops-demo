# UC function definitions and execute_python_code used by the agent.

def execute_python_code(code: str) -> str:
    """
    Executes the given python code and returns its stdout.
    Remember the code should print the final result to stdout.

    Args:
      code: Python code to execute. Remember to print the final result to stdout.
    """
    import sys
    from io import StringIO

    stdout = StringIO()
    sys.stdout = stdout
    try:
        exec(code)
        return stdout.getvalue()
    except Exception as e:
        if "Spark" in str(e):
            return f"Python code execution failed: {e}. Databricks specific code is not allowed."
        if "pyspark" in str(e):
            return f"Python code execution failed: {e}. Databricks specific code is not allowed."
        if "dbutils" in str(e):
            return f"Python code execution failed: {e}. Databricks specific code is not allowed."
        else:
            return f"Python code execution failed: {e}. Use simple code + try again."


ask_ai_function = """CREATE OR REPLACE FUNCTION {ask_ai_function_name}(question STRING COMMENT 'question to ask')
RETURNS STRING
COMMENT 'answer the question using chosen model'
RETURN SELECT ai_gen(question)
"""

summarization_function = """CREATE OR REPLACE FUNCTION {summarization_function_name}(text STRING COMMENT 'content to parse', max_words INT COMMENT 'max number of words in the response, must be non-negative integer, if set to 0 then no limit')
RETURNS STRING
COMMENT 'summarize the content and limit response to max_words'
RETURN SELECT ai_summarize(text, max_words)
"""

translate_function = """CREATE OR REPLACE FUNCTION {translate_function_name}(content STRING COMMENT 'content to translate', language STRING COMMENT 'target language')
RETURNS STRING
COMMENT 'translate the content to target language, currently only english <-> spanish translation is supported'
RETURN SELECT ai_translate(content, language)
"""

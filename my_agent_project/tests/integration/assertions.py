"""
Helper functions and assertions for Databricks Agent integration tests
"""
import inspect
from typing import List, Dict, Set, Any, Optional, Mapping, Iterator


def validate_uc_function_result(result, function_name: str):
    """
    Validate Unity Catalog function execution result
    """
    if result is None:
        raise AssertionError(f"Function {function_name} returned None")

    if not hasattr(result, 'value'):
        raise AssertionError(f"Function {function_name} result missing value attribute")
    return
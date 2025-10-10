from mlflow.genai.scorers import Scorer
from mlflow.entities import Feedback, AssessmentSource, SpanType
from mlflow.entities.span import SpanAttributeKey
from mlflow.genai.judges import custom_prompt_judge
from typing import Optional, Any, Dict, List, Union
import mlflow
import json

tool_requirement_prompt = """
Analyze the user's input and determine if the specified tool is required to properly answer their question.

User Input: {inputs}

Tool Name: {tool_name}
Tool Description: {tool_description}

Consider:
- Does the user's question require the specific capability this tool provides?
- Would the question be incomplete or incorrect without using this tool?

Respond with:
- [[required]] if the tool is necessary to properly answer the question
- [[not_required]] if the question can be answered adequately without this tool
"""

class ToolUsageScorer(Scorer):
    name: str = "tool_usage"
    
    def determine_required_tools(self, user_input: str, tools: Dict[str, str]) -> Dict[str, bool]:
        """
        use LLM judges to determine which tools should be required for the given input.
        returns: {tool_name: bool} where bool is whether the tool is required.
        """
        required_tools = {}
        for tool_name, tool_description in tools.items():
            judge = custom_prompt_judge(
                name=f"{tool_name}_requirement_judge",
                prompt_template=tool_requirement_prompt,
                numeric_values={"required": 1.0, "not_required": 0.0}
            )
            result = judge(
                inputs=user_input,
                tool_name=tool_name,
                tool_description=tool_description
            )
            required_tools[tool_name] = result.value == 1.0
        return required_tools

    def extract_used_tools_from_trace(self, trace: mlflow.entities.Trace) -> List[Dict[str, Any]]:
        """
        get tools used from trace. 
        """
        tools = []
        
        spans = trace.search_spans(span_type=SpanType.TOOL)
        for span in spans: 
            messages = span.get_attribute(SpanAttributeKey.OUTPUTS)
            content = json.loads(messages['content'])
            t = {"tool_call_id": messages['tool_call_id'], 
                "tool_name": messages['name'], 
                "tool_response": content['value'], 
                "tool_status": messages['status']}
            
            tools.append(t)
        return tools
    
    def interpret_tool_call_response(self, tool_name: str, tool_call_response: str, tool_status: str) -> str:
        """
        helper function to interpret the tool call response. 
        """
        if "execute_python_code" in tool_name:
            if "Python code execution failed" in tool_call_response:
                return "failed"
            else:
                return "success"
        else:
            return tool_status

        return tool_call_response
    

    def compare_tool_usage(self, required_tools: Dict[str, bool], used_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        return tool usage
        """
        required_tools = [name for name, required in required_tools.items() if required]
        correctly_used_tools = []
        failed_required_tools = []
        incorrectly_used_tools = []

        for tool in used_tools:
            if tool['tool_name'] in required_tools:
                correctly_used_tools.append(tool)
                required_tools.remove(tool['tool_name'])
                response = self.interpret_tool_call_response(tool['tool_name'], tool['tool_response'], tool['tool_status'])
                if response != "success":
                    failed_required_tools.append(tool)
            else:
                incorrectly_used_tools.append(tool)
                
        return {
            "correctly_used_tools": correctly_used_tools, # used and required ! :) 
            "incorrectly_used_tools": incorrectly_used_tools, # used but not required
            "failed_required_tools": failed_required_tools, # required but response was not successful
            "missing_required_tools": required_tools # required but not used
        }
            
    
    def __call__(
        self, 
        *, 
        inputs: Optional[dict[str, Any]],
        outputs: Optional[Any],
        expectations: Optional[dict[str, Any]], 
        trace: Optional[mlflow.entities.Trace]
    ) -> Feedback:
        """
        Custom scorer function for tool usage evaluation.
        """

        tools = {
            "execute_python_code": "Executes Python code for calculations, data analysis, and computations",
            "summarize": "Summarizes long text content using an LLM",
            "translate": "Translates text between different languages using an LLM", 
            "vector_search_retriever": "Retrieves relevant Databricks documentation from a vector search index"
        }

        user_input = inputs['question']
        
        required_tools = self.determine_required_tools(user_input, tools)
        used_tools = self.extract_used_tools_from_trace(trace)
        
        tool_usage_metrics = self.compare_tool_usage(required_tools, used_tools)

        
        mistakes = len(tool_usage_metrics["incorrectly_used_tools"]) + len(tool_usage_metrics["failed_required_tools"]) + len(tool_usage_metrics["missing_required_tools"])
        
        if mistakes == 0: 
            return Feedback(
                value=True,
                rationale="Used required tools properly. No feedback needed.",
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="tool_usage_scorer")
                )

        else: 
            return Feedback(
                value=False,
                rationale="Incorrectly used tools. Check metadata for more information. ",
                source=AssessmentSource(source_type="LLM_JUDGE", source_id="tool_usage_scorer")
                )
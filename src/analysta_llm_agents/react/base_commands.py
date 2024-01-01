
# Copyright (c) 2023 Artem Rozumenko

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Optional
from time import sleep
from langchain.schema import HumanMessage
from openai import RateLimitError
from .constants import analyze_prompt

def remember_last_message(ctx: Any, message: Optional[str]=''):
    """Store results of previous command in long term memory to persist between sessions
    Args:
        message (str): Message that describes what was stored
    """
    ctx.shared_memory.append({
        "role": "system", 
        "content": f"What is stored: {message}\n\nData: {ctx.last_message}"
    })
    return {"result": "Last message remembered"}


def complete_task(result: str) -> str:
    """Mark task as completed
    Args:
        result (str): "Result of task"
    """
    return {"result": result, "done": True}


def ask_llm(ctx: Any, hypothesis: str, data_for_analysis: str) -> str:
    """Ask LLM to help with task, make sure to provide data to help LLM to understand the task, scope and expected result
    Args:
        hypothesis (str): "Prompt for LLM"
        data_for_analysis (str): "Data for analysis"
    """
    predict_message = [ HumanMessage(content=analyze_prompt.format(hypothesis=hypothesis, data=data_for_analysis)) ]
    try:
        llm_reponse = ctx.llm(predict_message).content
    except RateLimitError as e:
        sleep(60)
        llm_reponse = ctx.llm(predict_message).content
    return {"result": llm_reponse}

__all__ = [
    remember_last_message,
    complete_task,
    ask_llm
]
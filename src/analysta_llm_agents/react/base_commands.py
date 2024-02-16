
#    Copyright 2023 Artem Rozumenko

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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


def complete_task(ctx: Any, result: str) -> str:
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

def ask_user(ctx: Any, question: str):
    """ Ask user a question and wait for input
    Agrs:
        question (str): question to be asked
    """
    pass


__all__ = [
    remember_last_message,
    complete_task,
    ask_llm,
    ask_user
]
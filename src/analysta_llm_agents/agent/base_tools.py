#
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
from langchain.schema import HumanMessage, AIMessage
from openai import RateLimitError
from .defaults import analyze_prompt, in_convesation_prompt
import logging

logger = logging.getLogger(__name__)


def ask_llm_within_conversation(ctx: Any, cid: str, role:str, task: str, query_to_search: str, expected_result:str) -> str:
    """Ask large languale model within context of conversation with all short-term memory and search in long-term memory
    
    Args:
        role (str): "Role of LLM"
        task (str): "Task to solve"
        query_to_search (str): "Request to search in long-term memory"
        expected_result (str): "What is expected as a result of the task"
    """
    messages = ctx.short_term_memory.copy()
    _messages = []
    if query_to_search and cid:
        logger.info("Conversation ID: " + cid)
        context = ctx.memory.search(query_to_search, cid)
        for message in context:
            _messages.append(HumanMessage(content=message["content"]))
    for message in messages:
        if message["role"] == "user":
            _messages.append(HumanMessage(content=message["content"]))
        else:
            _messages.append(AIMessage(content=message["content"]))
    
    predict_message = HumanMessage(content=in_convesation_prompt.format(
        role=role, task=task, expected_result=expected_result))
    
    _messages.append(predict_message)
    try:
        llm_reponse = ctx.llm.invoke(_messages).content
    except RateLimitError as e:
        sleep(60)
        llm_reponse = ctx.llm.invoke(_messages).content
    
    logger.info(f"LLM response: {llm_reponse}")
    
    return llm_reponse


def ask_llm_outside_conversation(ctx: Any, role:str, task: str, data_for_analysis: str, expected_result:str) -> str:
    """Ask large languale model to generate the response based on provided data. 
    the call is made outside of conversation, neither short-term nor long-term memory is used.
    
    Args:
        role (str): "Role of LLM"
        task (str): "Task to solve"
        data_for_analysis (str): "Data to use as context"
        expected_result (str): "What is expected as a result of the task"
    """
    predict_message = [ HumanMessage(content=analyze_prompt.format(
        role=role, task=task, data=data_for_analysis, 
        expected_result=expected_result
        ))]
    try:
        llm_reponse = ctx.llm.invoke(predict_message).content
    except RateLimitError as e:
        sleep(60)
        llm_reponse = ctx.llm.invoke(predict_message).content
    
    logger.info(f"LLM response: {llm_reponse}")
    
    return llm_reponse

def ask_user(ctx: Any, question: str): 
    """ Ask user a question and wait for input
    Agrs:
        question (str): question to be asked
    """
    pass

def complete_task(ctx: Any, result: str) -> str:
    """Mark task as completed
    Args:
        result (str): "Result of task"
    """
    pass


__all__ = [
    ask_llm_within_conversation,
    ask_llm_outside_conversation,
    complete_task,
    ask_user
]
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

import logging
from typing import Optional

from time import sleep
from tiktoken import get_encoding, encoding_for_model

from langchain import hub
from langchain.llms import __getattr__ as get_llm, __all__ as llms  # pylint: disable=E0401
from langchain.chat_models import __all__ as chat_models  # pylint: disable=E0401
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from openai import RateLimitError

from ..tools.context import Context
from ..tools.utils import unpack_json

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self, 
                 repo: Optional[str]=None, 
                 api_key: Optional[str]=None, 
                 agent_prompt: Optional[str]=None,
                 model_type: Optional[str]=None,
                 model_params: Optional[dict]=None, 
                 response_format: str="",
                 persist_messages: bool=False,
                 ctx: Optional[Context]=None):
        if ctx is None:
            self.ctx = Context()
        else:
            self.ctx = ctx
        self.ctx.last_message = ''
        self.agent_prompt = ''
        self.ctx.llm = self.get_model(model_type, model_params)
        self.model_name = model_params.get("model_name")
        self.response_format = response_format
        self.persist_messages = persist_messages
        
        if agent_prompt:
            self.prompt = agent_prompt
        elif repo:
            self.prompt = hub.pull(repo, api_key=api_key)
        else:
            raise RuntimeError("You must specify either repo or agent_prompt")
        self.agent_messages = []
        
        self.reset()
    
    
    @property
    def prompt(self):
        return self.agent_prompt
    
    @prompt.setter
    def prompt(self, value):
        self.agent_prompt = value.format(response_format=self.response_format)
    
    def get_model(self, model_type: str, model_params: dict):
        """ Get LLM or ChatLLM """
        if model_type is None:
            return None
        if model_type in llms:
            return get_llm(model_type)(**model_params)
        elif model_type in chat_models:
            model = getattr(__import__("langchain.chat_models", fromlist=[model_type]), model_type)
            return model(**model_params)
        raise RuntimeError(f"Unknown model type: {model_type}")

    def calculate_tokens(self):
        """ Calculate tokens for the prompt """
        try:
            encoding = encoding_for_model(self.model_name)
        except:
            encoding = get_encoding("cl100k_base")
        for message in self.agent_messages:
            self.ctx.input_tokens += len(encoding.encode(message["content"]))

    def count_tokens_from_response(self, response: str):
        """ Count tokens from response """
        try:
            encoding = encoding_for_model(self.model_name)
        except:
            encoding = get_encoding("cl100k_base")
        self.ctx.output_tokens += len(encoding.encode(response))
    
    
    def construct_messages(self, messages:Optional[list]):
        """ Construct messages """
        _messages = []
        self.calculate_tokens()
        if not messages:
            messages = self.agent_messages
        for message in messages:
            if message["role"] == "system":
                _messages.append(SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                _messages.append(HumanMessage(content=message["content"]))
            else:
                _messages.append(AIMessage(content=message["content"]))
        logger.debug(f"Constructed messages: {_messages}")
        return _messages
    
    def reset(self):
        self.agent_messages = []
        self.agent_messages.append({
            "role": "system",
            "content": self.prompt
        })
    
    def start(self, task: str):
        self.agent_messages.append({"role": "user", "content": task})
        yield from self.process_reponse()
    
    def get_response(self, messages: Optional[list]=None):
        try:
            response = self.ctx.llm(self.construct_messages(messages)).content
        except RateLimitError:
            sleep(60)
            response = self.ctx.llm(self.construct_messages(messages)).content
        self.count_tokens_from_response(response)
        try:
             json_response = unpack_json(response.replace("\n", ""))
             return json_response
        except:
            json_response = None
            return response
    
    def process_reponse(self):
        yield self.get_response()
        self.agent_messages = [] # Clear memory from previous tasks
        
        

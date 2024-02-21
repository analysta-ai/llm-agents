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


import logging
import json
import jsonpath_ng as jsonpath
from time import sleep

from typing import Any, Optional
from tiktoken import get_encoding, encoding_for_model
from openai import RateLimitError
from traceback import format_exc
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from analysta_index.interfaces.llm_processor import get_vectorstore, get_embeddings, get_model

from ..tools.context import Context
from ..tools.utils import unpack_json, MaxRetriesExceededError
from .base_tools import __all__ as base_tools
from .defaults import response_format, default_prompt
from .memory import Memory

logger = logging.getLogger(__name__)

class Agent:
    def __init__(
        self,
        agent_prompt: str, # this one should describe Role, Skills and Purpose of the agent
        llm_model_type: str,
        llm_model_params: dict, 
        short_term_memory_limit: int,
        embedding_model: Optional[str],
        embedding_model_params: Optional[dict],
        vectorstore: Optional[str],
        vectorstore_params: Optional[dict],
        text_response_format: Optional[dict] = response_format,
        message_key: Optional[str]="thoughts.text", # jsonpath to the message that should be displayed to user
        tool_key: Optional[str]='tool', # jsonpath to the command that should be executed
        contractor_key: Optional[str]='agent', # jsonpath to the contractor that should be executed
        plan_key:Optional[str]='thoughts.plan', # jsonpath to the plan that should be executed
        agent_constraints: Optional[str]='',
        datasources: Optional[list]=[],
        tools: Optional[list]=[],
        contractors: Optional[list]=[],
        max_retries: int=3,
        ctx: Optional[Any]=None
    ):
        if ctx is None:
            self.ctx = Context()
        else:
            self.ctx = ctx
        self.max_retries = max_retries
        self.text_response_format = text_response_format
        if not self.ctx.is_set("memory"):
            if vectorstore and embedding_model:
                self.embedding = get_embeddings(embedding_model, embedding_model_params)
                self.vectorstore = get_vectorstore(vectorstore, vectorstore_params, embedding_func=self.embedding)        
                self.ctx.memory = Memory(self.vectorstore, self.embedding)
        
        if not self.ctx.is_set("short_term_memory"):
            self.ctx.short_term_memory = []
        if not self.ctx.is_set("input_tokens"):
            self.ctx.input_tokens = 0
        if not self.ctx.is_set("output_tokens"):
            self.ctx.output_tokens = 0
            
        self.short_term_memory_limit = short_term_memory_limit
        
        agent_constraints += f"\n - Your short term memory is limited to {short_term_memory_limit} tokens, prefer not to be very chatty"
        agent_constraints += '\n - You have long term memory that stores stores information about the tasks you have solved, you can ask summary from it using "remind" tool'
        
        if not self.ctx.is_set("llm"):
            self.ctx.llm = get_model(llm_model_type, llm_model_params)
        
        try:
            self.encoding = encoding_for_model(self.model_name)
        except:
            self.encoding = get_encoding("cl100k_base")
        
        self.tools = self._get_tools(base_tools + tools)
        self.contractors = self._get_contractors(contractors)
        self.datasources = self._get_datasources(datasources)
        self.agent_prompt = self._agent_prompt(agent_prompt, agent_constraints, text_response_format)
        self.message_key = jsonpath.parse(message_key)
        self.tool_response = None
        self.contractor_key = None
        self.plan_key = None
        if tool_key:
            self.tool_response = jsonpath.parse(tool_key)
        if contractor_key:
            self.contractor_key = jsonpath.parse(contractor_key)
        if plan_key:
            self.plan_key = jsonpath.parse(plan_key)
        # this one will be used to correct behavior of the agent, like response format
        self.temporary_messages = [] 
        print(self.agent_prompt)
        # This will be for long term memory, where presumably 
        # we will search for relevant informatiom for the task
        
    def _agent_prompt(self, agent_prompt: str, agent_constraints: str, response_formant: json):
        instruments = ""
        if self.tools:
            instruments += f"\n\n{self.str_tools}"
        if self.contractors:
            instruments += f"\n\n{self.str_contractors}"
        if self.datasources:
            instruments += f"\n\n{self.str_datasources}"
        if self.tools and self.contractors:
            agent_constraints += '\n - Exclusively use the tools and contractors listed in double quotes e.g. "tool1" or "contractor1"'
            agent_constraints += '\n - You can use either one tool or one contactor at a time, not both'
        elif self.tools and not self.contractors:
            agent_constraints += '\n - Exclusively use the tools listed in double quotes e.g. "tool1"'
        elif not self.tools and self.contractors:
            # this is hypothetical as there are default tools for the agent
            agent_constraints += '\n - Exclusively use the contractors listed in double quotes e.g. "contractor1"'
        if self.datasources:
            agent_constraints += '\n - You have additional knowledge stored with datasources, you can search for it using "searchDatasource" tool'
        prompt = default_prompt.format(agent_prompt=agent_prompt, 
                                     instruments=instruments, 
                                     agent_constraints=agent_constraints,
                                     response_format=json.dumps(response_formant, indent=2))    

        return {"role": "system", "content": prompt, "tokens": len(self.encoding.encode(prompt))}
    
    @property
    def str_tools(self):
        return "Tools:\n" + "\n".join([tool["__repr__"] for tool in self.tools])

    @property
    def str_contractors(self):
        return "Contractors:\n" + "\n".join([contractor["__repr__"] for contractor in self.contractors])

    @property
    def str_datasources(self):
        return "Datasources:\n" + "\n".join([datasource["__repr__"] for datasource in self.datasources])
    
    def _get_tools(self, functions: list):
        """Create list of tools from functions docstrings"""
        result = []
        for func in functions:
            repr_data = func.__doc__.replace("\n", "").replace("    ", " ").split("Args:")
            if len(repr_data) == 1:
                repr_data.append("None")
            repr_data = f' - {repr_data[0]}: func: "{func.__name__}", args: {repr_data[1]}'
            result.append({
                "name": func.__name__,
                "__repr__": repr_data,
                "callable": func
            })
        logger.debug(f"Tools: {result}")
        return result
    
    def _get_contractors(self, contractors_list: list[Any]):
        """Create list of contractors for agent"""
        result = []
        for contractor in contractors_list:
            repr_data = f' - {contractor.__description__}: name: "{contractor.__name__}", args: "task": (str): "Task for agent to solve"'
            result.append({
                "name": contractor.__name__,
                "__repr__": repr_data,
                "callable": contractor
            })
        return result

    def _get_datasources(self, datasources: list[Any]):
        """Create list of datasources for agent"""
        result = []
        for datasource in datasources:
            repr_data = f' - {datasource.__description__}: name: "{datasource.__name__}", args: "task": (str): "Task for agent to solve"'
            result.append({
                "name": datasource.__name__,
                "__repr__": repr_data,
                "callable": datasource
            })
        return result

    def _get_callable_by_name(self, name: str, t: str):
        """Get callable from list of either tools or contractors"""
        ref = {
            'tool': {tool["name"]: tool["callable"] for tool in self.tools},
            'contractor': {contractor["name"]: contractor["callable"] for contractor in self.contractors},
            'datasource': {datasource["name"]: datasource["callable"] for datasource in self.datasources}
        }
        return ref[t].get(name)

    def add_to_short_term(self, role:str, message:str, conversation_id: str):
        self.ctx.short_term_memory.append({"role": role, "content": message, "tokens": len(self.encoding.encode(f"{role}: {message}"))})
        total_length = sum([message["tokens"] for message in self.ctx.short_term_memory])
        if total_length > self.short_term_memory_limit*0.9:
            messages = []
            user_message = 0
            short_term_memory_cut = 0
            for index, message in enumerate(self.ctx.short_term_memory):
                if message["role"] == "user" and user_message == 1:
                    short_term_memory_cut = index-1
                    break
                if message["role"] == "user":
                    user_message = 1
                messages.append(message)
            self.ctx.short_term_memory = self.ctx.short_term_memory[short_term_memory_cut:]
            document = "\n\n".join([f"{message['role']}: {message['content']}" for message in messages])
            self.ctx.memory.store(document, conversation_id, len(self.encoding.encode(document)))
    
    def calculate_tokens(self):
        """ Calculate tokens for the prompt """
        self.ctx.input_tokens += self.agent_prompt['tokens']
        print("Short Term Memory")
        print(self.ctx.short_term_memory)
        self.ctx.input_tokens += sum([message["tokens"] for message in self.ctx.short_term_memory])
    
    def count_tokens_from_response(self, response: str):
        """ Count tokens from response """
        self.ctx.output_tokens += len(self.encoding.encode(response))
    
    def construct_messages(self, messages:Optional[list]):
        """ Construct messages """
        _messages = []
        self.calculate_tokens()
        if not messages:
            messages = self.ctx.short_term_memory.copy()
            _messages.append(SystemMessage(content=self.agent_prompt['content']))
        if self.temporary_messages:
            messages += self.temporary_messages
        self.temporary_messages = []
        for message in messages:
            if message["role"] == "user":
                _messages.append(HumanMessage(content=message["content"]))
            else:
                _messages.append(AIMessage(content=message["content"]))

        logger.debug(f"Constructed messages: {_messages}")
        return _messages
    
    def get_response(self, messages: Optional[list]=None):
        try:
            response = self.ctx.llm.invoke(self.construct_messages(messages)).content
        except RateLimitError:
            sleep(60)
            response = self.ctx.llm.invoke(self.construct_messages(messages)).content
        self.count_tokens_from_response(response)
        try:
             return unpack_json(response.replace("\n", ""))
        except:
            return response
    
    def _pre_process(self, conversation_id, retry: int=0):
        json_response = self.get_response()
        logger.info(f"ReactAgent _pre_process: {json.dumps(json_response, indent=2)}")
        if retry >= self.max_retries:
            raise MaxRetriesExceededError(json_response)
        if isinstance(json_response, str):
            logging.error(f"Failed to get response: {json_response}")
            self.temporary_messages.append({
                "role": "user", 
                "content": f"""ERROR: RESPONSE FORMAT IS INCORRECT
The response may have a great data, format response to the required JSON strcuture. 
Expected format: {self.text_response_format}

Recieved data: {json_response}"""})
            return self._pre_process(conversation_id, retry=retry+1)
        tool_message = None
        contractor_message = None
        user_message = "\n\n".join([match.value for match in self.message_key.find(json_response)])    
        if not user_message:
            logger.error(f"User message on JSONPATH {str(self.message_key)} was not found in response. Try again")
            self.temporary_messages.append({"role": "user", "content": f"User message on JSONPATH {str(self.message_key)} was not found in response. Try again"})
            return self._pre_process(conversation_id, retry=retry+1)
        if self.tool_response:
            tool_message = [match.value for match in self.tool_response.find(json_response)]
            logger.info(f"Tool message from _pre_process: {tool_message}")
            if len(tool_message) == 1:
                tool_message = tool_message[0]
            elif len(tool_message) > 1:
                logger.error("Multiple tool messages found in response, only one allowed. Try again")
                self.temporary_messages.append({"role": "user", "content": f"Multiple tool messages found in response, only one allowed. Try again"})
                return self._pre_process(conversation_id, retry=retry+1)
        if self.contractor_key:
            contractor_message = [match.value for match in self.contractor_key.find(json_response)]
            if len(contractor_message) == 1:
                contractor_message = contractor_message[0]
            elif len(contractor_message) > 1:
                logger.error("Multiple contractor messages found in response, only one allowed. Try again")
                self.temporary_messages.append({"role": "user", "content": f"Multiple contractor messages found in response, only one allowed. Try again"})
                return self._pre_process(conversation_id, retry=retry+1)
        if tool_message and contractor_message:
            logger.error("Both tool and contractor messages found in response, only one allowed. Try again")
            self.temporary_messages.append({"role": "user", "content": f"Both tool and contractor messages found in response, only one allowed. Try again"})
            return self._pre_process(conversation_id, retry=retry+1)
        plan = ''
        if self.plan_key:
            plan = str([match.value for match in self.plan_key.find(json_response)])
            
        self.add_to_short_term("assistant", f"{user_message}\n\n{plan}", conversation_id )
        return user_message, tool_message, contractor_message
    
    def process_command(self, command: str, command_args: dict, conversation_id: str):
        logger.info(f"Agent process_command: {command.__name__},  args: {command_args}")
        try:
            if command.__name__ == 'ask_llm_within_conversation':
                # TODO: this is somehow weird, as anything that require 
                # short term memory should require conversation_id
                res = command(self.ctx, conversation_id, **command_args)
            else:
                res = command(self.ctx, **command_args)
            self.add_to_short_term("assistant", res, conversation_id)
            self.ctx.last_message = res
        except KeyError as e:
            logger.error(format_exc())
            self.temporary_messages.append({
                "role": "user", 
                "content": f"ERROR: Some colums do not exist, verify the names. ${str(e)}. Use get_column_names to see all columns."
            })
        except ValueError as e:
            logger.error(format_exc())
            self.temporary_messages.append({
                "role": "user", 
                "content": f"ERROR: Data extraction error: {str(e)}"
            })
        except Exception as e:
            logger.error(format_exc())
            self.temporary_messages.append({
                "role": "user", 
                "content":
                    f"ERROR: {str(e)}"
            })
    
    def process_response(self, conversation_id):
        try:
            user_message, tool_message, contractor_message = self._pre_process(conversation_id)
            logger.info(f"User message: {user_message}")
            logger.info(f"Tool message: {tool_message}")
            logger.info(f"Contractor message: {contractor_message}")
            
            yield user_message
            
            if tool_message:
                if tool_message['name'] ==  'ask_user':
                    yield tool_message['args']['question']
                    self.add_to_short_term("assistant", tool_message['args']['question'], conversation_id)
                    return
                elif tool_message['name'] == 'complete_task':
                    yield tool_message['args']['result']
                    self.add_to_short_term("assistant", tool_message['args']['result'], conversation_id)
                    self.ctx.last_message = tool_message['args']['result']
                    return 
                else:
                    # TODO: Add specific command to search for data in datasources
                    command = self._get_callable_by_name(tool_message['name'], 'tool')
                    if command:
                        self.process_command(command, tool_message.get('args', {}), conversation_id)
                    else:
                        self.temporary_messages.append(
                            {"role": "user", "content": "ERROR: Unknown command. Check the name of command"}
                        )
                    yield from self.process_response(conversation_id)
            elif contractor_message:
                if contractor_message['name'] == 'complete_task':
                    yield contractor_message['args']['result']
                    self.ctx.last_message = contractor_message['args']['result']
                    self.add_to_short_term("assistant", tool_message['args']['result'], conversation_id)
                    self.ctx.last_message = tool_message['args']['result']
                    return
                else:
                    contractor = self._get_callable_by_name(tool_message['name'], 'contractor')
                    if contractor:
                        for message in contractor.start(contractor_message['args']['task'], conversation_id):
                            yield message
                    else:
                        self.temporary_messages.append(
                            {"role": "user", "content": "ERROR: Unknown contractor. Check the name of contractor"}
                        )
                yield from self.process_response(conversation_id)
        except MaxRetriesExceededError as e:
            yield e.message   
        
    def start(self, task: str, conversation_id: str):
        self.clear_on_start = True
        logger.debug("MasterAgent start: %s", task)
        self.add_to_short_term("user", task, conversation_id)
        yield from self.process_response(conversation_id)
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
from langchain import hub
from json import dumps

from ..tools.context import Context
from ..react.agent import ReactAgent
from ..tools.utils import unpack_json
from .constants import agent_response_format, gk_response_format

class MasterAgent(ReactAgent):
    def __init__(self, 
                 repo: Optional[str]=None, 
                 api_key: Optional[str]=None, 
                 agent_prompt: Optional[str]=None,
                 agents: Optional[list]=None,
                 model_type: Optional[str]=None,
                 model_params: Optional[dict]=None, 
                 response_format: str=agent_response_format,
                 gatekeeper_repo: Optional[str]=None,  # This one needed if we want somehow validate user input
                 gatekeeper_reponse_format: Optional[str]=gk_response_format,
                 ctx: Optional[Context]=None):
        
        if ctx is None:
            ctx = Context()
        
        self.agents = self.get_agents(ctx, agents)
        self.gatekeeper = None
        if gatekeeper_repo:
            self.gatekeeper = hub.pull(gatekeeper_repo, api_key=api_key)
            self.gk_response_format = gatekeeper_reponse_format
        
        super().__init__(repo=repo, api_key=api_key, agent_prompt=agent_prompt, 
                         actions=[], model_type=model_type, model_params=model_params, 
                         response_format=response_format, ctx=ctx)
    
    @property
    def str_agents(self):
        agents_repr = []
        for agent in self.agents:
            agents_repr.append(agent["__repr__"])
        return "\n".join(agents_repr)
    
    @property
    def prompt(self):
        return self.agent_prompt

    @prompt.setter
    def prompt(self, value):
        self.agent_prompt = value.format(commands=self.str_commands, agents=self.str_agents, response_format=self.response_format)
    
    
    def get_agents(self, ctx, agents_config: list[Any]):
        """Create list of tools from functions docstrings"""
        agents = []
        for index, agent in enumerate(agents_config):
            agent_instance = agent(ctx)
            repr_data = f'{index}. {agent_instance.__description__}: name: "{agent_instance.__name__}", args: "task": (str): "Task for agent to solve"'
            agents.append({
                "name": agent_instance.__name__,
                "__repr__": repr_data,
                "agent": agent_instance
            })
        return agents
    
    def get_agent_by_name(self, name: str):
        """Get function by name from list of tools"""
        for agent in self.agents:
            if agent["name"] == name:
                return agent["agent"]
        return None
    
    def gatekeeper_check(self, question: str, gk_data: str):
        gk_messages = [{
            "role": "system",
            "content": self.gatekeeper.format(response_format=self.gk_response_format)
        }]
        for message in self.ctx.shared_memory:
            if message not in gk_messages:
                gk_messages.append(message)
        gk_messages.append({
            "role": "user",
            "content": f"Question: {question}\n\Available Data: {gk_data}"
        })
        return self.get_response(gk_messages)
        
    
    def start(self, task: str, clear: bool=True, gk_data: str=""):
        if clear:
            self.clear_messages()
        if gk_data and self.gatekeeper:
            result = self.gatekeeper_check(task, gk_data)
            if isinstance(result, dict):
                if not result.get("valid_question", False):
                    yield result["result"]
                    return
            else:
                yield result
                return
        print("[DEBUG] MasterAgent start: ", task)
        self.agent_messages.append({"role": "user", "content": f"Task: \n\n{task}"})
        yield from self.process_reponse()
    
    def highlights_from_agent(self):
        for message in self.ctx.shared_memory:
            if message not in self.agent_messages:
                self.agent_messages.append(message)
        print(self.agent_messages)

    def process_reponse(self):
        result, json_response, do_continue = self._pre_process_command()
        print("[DEBUG] MasterAgent:", dumps(json_response, indent=2))
        yield result
        if do_continue:
            return
        if json_response['command']['type'] == 'agent':
            agent = self.get_agent_by_name(json_response['command']['name'])
            print("[DEBUG] Agent: ", str(agent))
            if agent:
                for message in agent.start(json_response['command']['args']['task']):
                    yield message
                self.highlights_from_agent()
                self.agent_messages.append({"role": "system", "content": self.ctx.last_message})
            else:
                print("ERROR: Unknown agent. Check the name of agent")
                self.agent_messages.append({"role": "assistant", "content": "ERROR: Unknown agent. Check the name of agent"})
        elif json_response['command']['type'] == 'command':
            command, context_required = self.get_func_by_name(json_response["command"]["name"])
            if command:
                result, do_continue = self.process_command(command, json_response, context_required)
                if do_continue:
                    yield result
                    return 
            else:
                self.agent_messages.append({"role": "assistant", "content": "ERROR: Unknown command. Check the name of command"})
        elif json_response['command']['type'] == 'complete_task':
            yield json_response['command']['args']['result']
            return
        else:
            self.agent_messages.append({"role": "assistant", "content": "ERROR: Unknown command type. Check the type of command"})
        yield from self.process_reponse()

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

from typing import Optional
from json import dumps

from ..tools.context import Context
from ..base.agent import BaseAgent
from .constants import agent_response_format
from .base_commands import __all__ as base_actions

class ReactAgent(BaseAgent):
    def __init__(self, 
                 repo: Optional[str]=None,
                 api_key: Optional[str]=None, 
                 agent_prompt: Optional[str]=None,
                 actions: Optional[list]=None,
                 model_type: Optional[str]=None,
                 model_params: Optional[dict]=None, 
                 response_format: str=agent_response_format,
                 ctx: Optional[Context]=None):
        
        if ctx is None:
            ctx = Context()
            
        if actions:
            actions = actions + base_actions
        else:
            actions = base_actions
        
        self.tools = self.get_tools(actions)
        
        super().__init__(repo=repo, api_key=api_key, agent_prompt=agent_prompt, 
                         model_type=model_type, model_params=model_params, 
                         response_format=response_format, ctx=ctx)
    
    
    @property
    def str_commands(self):
        commands = []
        for tool in self.tools:
            commands.append(tool["__repr__"])
        # print (commands)
        return "\n".join(commands)
    
    @property
    def prompt(self):
        return self.agent_prompt

    @prompt.setter
    def prompt(self, value):
        self.agent_prompt = value.format(commands=self.str_commands, response_format=self.response_format)
    
    def get_tools(self, functions: list):
        """Create list of tools from functions docstrings"""
        tools = []
        for index, func in enumerate(functions):
            repr_data = func.__doc__.replace("\n", "").replace("    ", " ").split("Args:")
            if len(repr_data) == 1:
                repr_data.append("None")
            repr_data = f'{index}. {repr_data[0]}: func: "{func.__name__}", args: {repr_data[1]}'
            context_required = 'ctx' in list(func.__annotations__.keys())
            tools.append({
                "name": func.__name__,
                "__repr__": repr_data,
                "func": func,
                "context": context_required
            })
        # print(tools)
        return tools
    
    def get_func_by_name(self, name: str):
        """Get function by name from list of tools"""
        for tool in self.tools:
            if tool["name"] == name:
                return tool["func"], tool["context"]
        return None, False 

    
    def clear_messages(self):
        self.reset()    
        for message in self.ctx.shared_memory:
            self.agent_messages.append(message)
    
    def start(self, task: str, clear: bool=True):
        if clear:
            self.clear_messages()
        # print("[DEBUG] ReactAgent start: ", self.agent_messages)
        print("[DEBUG] ReactAgent start: ", task)
        # yield from super().start(task)
        self.agent_messages.append({"role": "user", "content": f"Task: \n\n{task}"})
        yield from self.process_reponse()
    
    def process_command(self, command: str, command_args: dict, context_required: bool=False):
        print("[DEBUG]: Command: ", command.__name__, "Context required: ", 
              context_required, "Args: ", command_args["command"].get("args",{}))
        print()
        if context_required:
            res = command(self.ctx, **command_args["command"].get("args",{}))
        else:
            res = command(**command_args["command"].get("args",{}))
        try:
            print("[DEBUG]: Command result: ", dumps(res, indent=2))
        except:
            pass
        keys = list(res.keys())
        self.agent_messages.append({"role": "assistant", "content": res['result']})
        self.ctx.last_message = res["result"]
        if 'done' in keys and res['done']:
            return res['result'], True
        keys.pop(keys.index('result'))
        if len(keys) > 0:
            for key in keys:
                print(key, res[key])
                self.ctx.__setattr__(key, res[key])
        return None, False

    def _pre_process_command(self, action_key: str="command"):
        json_response = self.get_response()
        # print("[DEBUG]: _pre_process_process_reponse :", dumps(json_response, indent=2))
        
        if isinstance(json_response, str):
            self.agent_messages.append({"role": "user", "content": "You failed to return response in expecteed format. Try again"})    
            return self._pre_process_command()
        try:
            result = json_response["thoughts"]["text"]
        except:
            self.agent_messages.append({"role": "user", "content": "You failed to return response in expecteed format. Try again"})    
            return self._pre_process_command()
        # try:
            # del json_response["thoughts"]["reasoning"]
            # del json_response["thoughts"]["criticism"]
        # except:
        #     pass
        if json_response.get(action_key) is None:
            print("[DEBUG]: No command in response", dumps(json_response, indent=2))
            return "Task is done", {}, True
        self.agent_messages.append({"role": "assistant", "content": dumps(json_response)})
        return result, json_response, False
        
    def process_reponse(self):
        result, json_response, do_continue = self._pre_process_command()
        # print("[DEBUG]: process_reponse :", dumps(json_response, indent=2))
        yield result
        if do_continue:
            return
        # TODO: This one need to be refactored
        command, context_required = self.get_func_by_name(json_response["command"]["name"])
        if command:
            result, do_continue = self.process_command(command, json_response, context_required)
            if do_continue:
                yield result
                return 
        else:
            self.agent_messages.append({"role": "assistant", "content": "ERROR: Unknown command. Check the name of command"})
        yield from self.process_reponse()

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

agent_response_format = """{
    "thoughts": {
        "text": "action summary to say to user, use bullets and tables to present data",
        "plan": "short bulleted, list that conveys long-term plan",
        "criticism": "constructive self-criticism",   
    }
    
    "command": {
        "type": "agent|command"
        "name": "command or agent name",
        "args": {
            "arg name": "value"
        }
    }
}"""

gk_response_format = """{
  "valid_question": true|false, 
  "result": "short explanation of decision"
}"""
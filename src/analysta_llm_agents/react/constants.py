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


# "reasoning": " crisp reasoning, preferably in 2-4 bullet points",

agent_response_format = """{
    "thoughts": {
        "text": "action summary to say to user, use bullets and tables to present data",
        "plan": "short bulleted, list that conveys long-term plan",
        "criticism": "constructive self-criticism",
        
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}"""


analyze_prompt = """This is the question:
{hypothesis}

Additional context:
{data}

Expected result: Provide exact answer, assuming that user can't execute code.
"""

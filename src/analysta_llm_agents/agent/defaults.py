
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


response_format = {
    "thoughts": {
        "text": "message to a user in crisp and clear business language",
        "plan": "short bulleted, list that conveys long-term plan",
        "criticism": "constructive self-criticism",
    },
    "tool": {
        "name": "tool name",
        "args": { "arg name": "value" }
    },
    "contractor": {
        "name": "contractor name",
        "args": { "arg name": "value" }
    }
}

analyze_prompt = """ Act as ane experienced {role} tasked to help user with:
{task}

Context you have in your possession:
{data}

Expected result: 
{expected_result}
"""

in_convesation_prompt = """ Act as ane experienced {role} tasked to help user with:
{task}

Expected result: 
{expected_result}
"""

default_prompt = """{agent_prompt}

Constraints:
{agent_constraints}
    
{instruments}

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

Respond only with JSON format as described below
{response_format}

Ensure the response contains only JSON and it could be parsed by Python json.loads
"""

weights = {
    'keywords': 0.2,
    'document_summary': 0.5,
    'data': 0.3
}

splitter_name='chunks'

splitter_params={
    'chunk_size': 2000,
    'chunk_overlap': 200,
    'autodetect_language': True,
    'kw_for_chunks': False,
}
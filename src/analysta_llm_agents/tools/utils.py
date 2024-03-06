# Copyright (c) 2023 Artem Rozumenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import re

def unpack_json(json_data, message_key=None):
    if (isinstance(json_data, str)):
        if '```json' in json_data:
            pattern = r'```json(.*)```'
            matches = re.findall(pattern, json_data, re.DOTALL)
            text = json_data.replace(f'{matches[0]}', '').replace('```json', '').replace('```', '').strip()
            res = json.loads(matches[0])
            if message_key and text:
                txt = "\n".join([match.value for match in message_key.find(res)])
                message_key.update(res, txt + text)
            return res
        return json.loads(json_data)
    elif (isinstance(json_data, dict)):
        return json_data
    else:
        raise ValueError("Wrong type of json_data")

class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of retries is exceeded"""
    
    def __init__(self, message="Maximum number of retries exceeded"):
        self.message = message
        super().__init__(self.message)
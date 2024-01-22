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

import functools

def tool(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            results_dict = {
                "result": "",
            }
            result = func(*args, **kwargs)
            if not isinstance(result, dict):
                results_dict["result"] = result
        except KeyError as e:
            results_dict["result"] = f"ERROR: Some colums do not exist, verify the names. ${str(e)}. Use get_column_names to see all columns."
        except ValueError as e:
            results_dict["result"] = f"ERROR: Data extraction error: {str(e)}"
        except Exception as e:
            results_dict["result"] = f"ERROR: {str(e)}"
        return results_dict
        
    return wrapper
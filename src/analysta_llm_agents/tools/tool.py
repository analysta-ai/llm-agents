# Copyright (c) 2023 Artem Rozumenko

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
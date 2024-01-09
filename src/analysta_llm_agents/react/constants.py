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

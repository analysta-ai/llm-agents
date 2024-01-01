import json

def unpack_json(json_data):
    if (isinstance(json_data, str)):
        if '```json' in json_data:
            json_data = json_data.replace('```json', '').replace('```', '')
            return json.loads(json_data)
        return json.loads(json_data)
    elif (isinstance(json_data, dict)):
        return json_data
    else:
        raise ValueError("Wrong type of json_data")
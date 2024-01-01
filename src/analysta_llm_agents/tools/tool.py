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
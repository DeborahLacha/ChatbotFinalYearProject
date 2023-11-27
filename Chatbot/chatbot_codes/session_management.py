from collections import defaultdict

SESSION_DATA = defaultdict(dict)

def get_session_dict(session_id):
    return SESSION_DATA[session_id]

def update_session_dict(session_id, context_var_name, context_var_value):
    session_dict = get_session_dict(session_id)
    session_dict[context_var_name] = context_var_value
# info we expect the LLM to return. can be expanded later (allergies, procedures, labs, vitals, etc.).

REQUIRED_TOP_KEYS = ["conditions", "medications"]

REQUIRED_CONDITION_KEYS = ["name"]
REQUIRED_MEDICATION_KEYS = ["name"]  

def validate_structured(obj):
    # must be a dict with the two top level keys
    if not isinstance(obj, dict):
        return False, "top level is not a dict"

    for k in REQUIRED_TOP_KEYS:
        if k not in obj:
            return False, "missing top level key: {}".format(k)

    # should contain lists of itemss
    if not isinstance(obj.get("conditions"), list):
        return False, "'conditions' is not a list"
    if not isinstance(obj.get("medications"), list):
        return False, "'medications' is not a list"

    # each condition and medication should be a dictionary and has name
    for i, c in enumerate(obj.get("conditions")):
        if not isinstance(c, dict):
            return False, "conditions[{}] is not an object".format(i)
        for req in REQUIRED_CONDITION_KEYS:
            if not c.get(req):
                return False, "conditions[{}] missing '{}'".format(i, req)

    for i, m in enumerate(obj.get("medications")):
        if not isinstance(m, dict):
            return False, "medications[{}] is not an object".format(i)
        for req in REQUIRED_MEDICATION_KEYS:
            if not m.get(req):
                return False, "medications[{}] missing '{}'".format(i, req)

    return True, "ok"

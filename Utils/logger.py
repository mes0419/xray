import datetime

log_config = True

def log(Tag, String):
    if log_config:
        print(datetime.datetime.now(tz=None),':',"[",Tag,']',String)

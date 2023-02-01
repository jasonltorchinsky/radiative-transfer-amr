from datetime import datetime

def print_msg(msg):
    """
    Prints the given message with the current time.
    """

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(('[{}]: {}').format(current_time, msg))

    return None

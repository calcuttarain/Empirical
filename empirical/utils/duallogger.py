class DualLogger:
    """Send text simultaneously to console and log file."""

    def __init__(self, filepath, stream):
        self.terminal = stream
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  
        self.log_file.write(message)
        self.log_file.flush() 

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

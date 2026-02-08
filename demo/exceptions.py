class HaltTraining(Exception):
    def __init__(self, context=None):
        self.context = context

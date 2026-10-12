"""Load a policy on its first action, after the simulator's initial reset."""


class DeferredPolicy:
    def __init__(self, factory, config):
        self.factory = factory
        self.config = config
        self.policy = None

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

    def eval(self, *args):
        if self.policy is None:
            print("Loading policy after the initial environment reset.", flush=True)
            self.policy = self.factory(self.config)
            self.policy.reset()
        return self.policy.eval(*args)

    def close(self):
        if self.policy is not None:
            self.policy.close()

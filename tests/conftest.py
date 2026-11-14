from hypothesis import settings

settings.register_profile("ci", derandomize=True, max_examples=200, deadline=None)
settings.load_profile("ci")

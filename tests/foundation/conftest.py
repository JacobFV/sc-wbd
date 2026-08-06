import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs that need the GPU and the corpus")

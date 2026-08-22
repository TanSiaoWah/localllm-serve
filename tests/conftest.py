import pytest


@pytest.fixture
def ticket():
    return {
        "subject": "Login issue",
        "message": "I cannot log into my account.",
    }
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dify_plugin.errors.model import CredentialsValidateFailedError  # noqa: E402

from provider.moonshot import MoonshotProvider  # noqa: E402


def _provider(model_ids):
    provider = MoonshotProvider.__new__(MoonshotProvider)
    provider.provider_schema = SimpleNamespace(
        models=[SimpleNamespace(model=model_id) for model_id in model_ids],
        provider="moonshot",
    )
    return provider


def _not_found(model: str) -> CredentialsValidateFailedError:
    return CredentialsValidateFailedError(
        "Credentials validation failed with status code 404 and response body "
        '{"error":{"message":"Not found the model %s or Permission denied",'
        '"type":"resource_not_found_error"}}' % model
    )


def test_falls_back_to_a_model_the_key_can_reach():
    # Regression for #3496: a key entitled only to newer kimi-k* models must
    # not be rejected just because the first predefined model (the legacy
    # moonshot-v1-8k default) 404s for it.
    provider = _provider(["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k3"])
    attempted = []

    def _validate(model, credentials):
        attempted.append(model)
        if model != "kimi-k3":
            raise _not_found(model)

    model_instance = Mock()
    model_instance.validate_credentials.side_effect = _validate
    provider.get_model_instance = Mock(return_value=model_instance)

    provider.validate_provider_credentials({"api_key": "key-with-only-kimi-k3"})

    assert attempted == ["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k3"]


def test_raises_last_error_when_no_model_is_reachable():
    provider = _provider(["moonshot-v1-8k", "moonshot-v1-32k"])

    def _validate(model, credentials):
        raise CredentialsValidateFailedError(f"401 invalid_authentication_error for {model}")

    model_instance = Mock()
    model_instance.validate_credentials.side_effect = _validate
    provider.get_model_instance = Mock(return_value=model_instance)

    with pytest.raises(CredentialsValidateFailedError, match="moonshot-v1-32k"):
        provider.validate_provider_credentials({"api_key": "bad-key"})


def test_stops_probing_on_first_success():
    provider = _provider(["moonshot-v1-8k", "moonshot-v1-32k", "kimi-k3"])
    attempted = []

    def _validate(model, credentials):
        attempted.append(model)

    model_instance = Mock()
    model_instance.validate_credentials.side_effect = _validate
    provider.get_model_instance = Mock(return_value=model_instance)

    provider.validate_provider_credentials({"api_key": "legacy-key"})

    assert attempted == ["moonshot-v1-8k"]

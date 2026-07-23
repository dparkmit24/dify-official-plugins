import logging
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin import ModelProvider

logger = logging.getLogger(__name__)


class MoonshotProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        Moonshot API keys are not all entitled to the same predefined models
        (legacy moonshot-v1-* vs. the newer kimi-k* lineup are gated per
        account/platform), so pinging a single hardcoded model 404s for keys
        that only have access to a different lineup. Probe the predefined
        models in their declared order and accept the key on the first one
        it can reach.

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        model_instance = self.get_model_instance(ModelType.LLM)
        last_error: CredentialsValidateFailedError | None = None
        for model_schema in self.provider_schema.models:
            try:
                model_instance.validate_credentials(model=model_schema.model, credentials=credentials)
                return
            except CredentialsValidateFailedError as ex:
                last_error = ex
            except Exception as ex:
                logger.exception(f"{self.get_provider_schema().provider} credentials validate failed")
                raise ex
        if last_error is not None:
            raise last_error

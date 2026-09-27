class AICoverError(Exception):
    """Base error shown to CLI users."""


class ConfigurationError(AICoverError):
    pass


class ExternalCommandError(AICoverError):
    def __init__(self, message: str, *, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code


class ArtifactError(AICoverError):
    pass

"""Project-specific exceptions."""


class TGADError(RuntimeError):
    """Base error for a benchmark configuration or data-layout failure."""


class DataValidationError(TGADError):
    """Raised when a benchmark dataset or annotation does not match the protocol."""


class BackendConfigurationError(TGADError):
    """Raised when a model backend cannot be configured safely."""

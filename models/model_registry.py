from typing import Dict, Type, List, Optional

from models.base_model import BaseModel
from utils.logger import get_model_logger

logger = get_model_logger("ModelRegistry")

class ModelRegistry:
    """
    A generic registry for all model architectures.
    Provides decorator-based registration and dynamic retrieval.
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class under a specific name.

        Args:
            name (str): The unique name to register the model as.
        """
        def wrapper(model_cls: Type[BaseModel]):
            if not issubclass(model_cls, BaseModel):
                logger.error(f"Cannot register '{name}': Must be a subclass of BaseModel.")
                raise TypeError(f"Model {model_cls.__name__} must inherit from BaseModel.")
                
            if name in cls._registry:
                logger.warning(f"Model '{name}' is already registered. Overwriting.")
                
            cls._registry[name] = model_cls
            logger.info(f"Registered model: '{name}' -> {model_cls.__name__}")
            return model_cls
            
        return wrapper

    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """
        Retrieves the model class associated with the given name.

        Args:
            name (str): The registered name of the model.

        Returns:
            Type[BaseModel]: The model class.
            
        Raises:
            KeyError: If the model name is not found in the registry.
        """
        if name not in cls._registry:
            logger.error(f"Model '{name}' not found in registry.")
            raise KeyError(f"Model '{name}' is not registered. Available models: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def remove(cls, name: str) -> None:
        """
        Removes a model from the registry.

        Args:
            name (str): The registered name to remove.
        """
        if name in cls._registry:
            del cls._registry[name]
            logger.info(f"Removed model '{name}' from registry.")

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Lists all registered model names.

        Returns:
            List[str]: List of registered model names.
        """
        return list(cls._registry.keys())


from models.base_model import BaseModel
from models.model_config import ModelConfig
from models.model_registry import ModelRegistry
from utils.logger import get_model_logger

logger = get_model_logger("ModelFactory")

class ModelFactory:
    """
    Factory for instantiating registered models based on configuration.
    """

    @staticmethod
    def create(model_name: str, config: ModelConfig) -> BaseModel:
        """
        Instantiates a model using the ModelRegistry and initializes it.

        Args:
            model_name (str): The registered name of the model to build.
            config (ModelConfig): The configuration to pass to the model.

        Returns:
            BaseModel: An instantiated, initialized, and device-mapped model.
        """
        logger.info(f"Factory creating model: '{model_name}'")
        
        # 1. Retrieve the model class from the registry
        model_cls = ModelRegistry.get_model_class(model_name)
        
        # 2. Instantiate the model
        model = model_cls(config)
        
        # 3. Apply standard BaseModel initializations
        model.initialize_weights()
        model.move_to_device()
        
        logger.info(f"Model '{model_name}' successfully created and initialized.")
        return model


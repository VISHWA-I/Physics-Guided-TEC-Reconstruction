from enum import Enum
import torch
from utils.logger import get_model_logger

logger = get_model_logger("CurriculumLearning")

class CurriculumStage(Enum):
    STAGE_1_QUIET = 1    # Kp <= 2, Dst > -30
    STAGE_2_MODERATE = 2 # Kp 3-5, Dst -30 to -80
    STAGE_3_STORM = 3    # Kp >= 6, Dst < -80
    STAGE_4_ALL = 4      # All data

class CurriculumLearning:
    """
    Manages the phased introduction of increasingly difficult space weather events.
    Prevents catastrophic forgetting by mastering quiet-time physics before attempting
    highly chaotic storm reconstructions.
    """
    
    def __init__(self, enable: bool = True, advance_patience: int = 5):
        self.enable = enable
        self.current_stage = CurriculumStage.STAGE_1_QUIET if enable else CurriculumStage.STAGE_4_ALL
        self.advance_patience = advance_patience
        self.epochs_without_improvement = 0
        self.best_val_loss = float('inf')
        logger.info(f"Curriculum Learning initialized. Starting at {self.current_stage.name}")

    def filter_batch(self, batch_data: dict) -> bool:
        """
        Determines if a batch should be trained on during the current curriculum stage.
        If returning False, the Trainer will `continue` and skip this batch.
        
        Assumes batch_data contains 'storm_feats' where Kp or Dst are accessible.
        (This is a placeholder simulation function until data pipelines are connected).
        """
        if not self.enable or self.current_stage == CurriculumStage.STAGE_4_ALL:
            return True
            
        # In a real scenario, we extract Kp and Dst from the storm_feats tensor.
        # For this architectural phase, we will simulate the filter as "Pass All" 
        # to ensure the architecture compiles and trains.
        # Actual implementation requires mapping indices of storm_feats to physical variables.
        return True

    def on_epoch_end(self, current_val_loss: float) -> None:
        """
        Evaluates whether to advance the curriculum stage based on validation plateaus.
        """
        if not self.enable or self.current_stage == CurriculumStage.STAGE_4_ALL:
            return
            
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        if self.epochs_without_improvement >= self.advance_patience:
            self.advance_stage()

    def advance_stage(self):
        if self.current_stage == CurriculumStage.STAGE_1_QUIET:
            self.current_stage = CurriculumStage.STAGE_2_MODERATE
        elif self.current_stage == CurriculumStage.STAGE_2_MODERATE:
            self.current_stage = CurriculumStage.STAGE_3_STORM
        elif self.current_stage == CurriculumStage.STAGE_3_STORM:
            self.current_stage = CurriculumStage.STAGE_4_ALL
            
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        logger.info(f"Curriculum Advanced! New Stage: {self.current_stage.name}")

"""
Base class for safety evaluation datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, List


class SafetyDataset(ABC):
    """
    Abstract base class for safety evaluation datasets.
    
    Each dataset should implement __iter__ to yield evaluation samples
    and optionally provide category information.
    """
    
    @abstractmethod
    def __iter__(self) -> Iterable[Dict]:
        """
        Iterate over the dataset samples.
        
        Yields:
            Dict with keys like:
                - id: Sample identifier
                - prompt_1: User input/prompt
                - prompt_2: Model response
                - media: Optional path to media file (image/audio/video)
                - media_type: Optional media type (image/audio/video)
                - label: Optional ground truth label
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    def get_categories(self) -> Optional[List[str]]:
        """
        Return the list of safety categories for this dataset.
        
        Returns:
            List of category names or None if not applicable.
        """
        return None


class SafetyEvaluator(ABC):
    """
    Abstract base class for safety evaluators.
    
    Evaluators assess whether content is safe or unsafe according
    to specified safety policies.
    """
    
    @abstractmethod
    def evaluate(
        self,
        prompt_1: str,
        prompt_2: str,
        *,
        categories: Optional[List] = None,
        **kwargs
    ) -> str:
        """
        Evaluate the safety of a conversation.
        
        Args:
            prompt_1: User input/prompt
            prompt_2: Model response
            categories: Optional list of safety categories to check
            **kwargs: Additional arguments (e.g., media, media_type)
            
        Returns:
            Evaluation result string (e.g., "safe" or "unsafe\nS1,S3")
        """
        pass

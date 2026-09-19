import litellm
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMAdvisor:
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo", model: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model or model_name
        self.available = bool(api_key and api_key.strip())
        
        if not self.available:
            logger.info("LLM api_key is empty. LLMAdvisor will be disabled.")
        else:
            logger.info(f"LLMAdvisor initialized with model {model_name}")

    def get_status(self) -> str:
        return "Active" if self.available else "Disabled"

    async def get_market_commentary(self, symbol: str, bias: str, rsi: float, adx: float, structure_summary: str) -> str:
        if not self.available:
            return 'LLM unavailable'
            
        prompt = (
            f"Please provide a brief market analysis for {symbol}.\n"
            f"Current Market Bias: {bias}\n"
            f"RSI: {rsi:.2f}\n"
            f"ADX: {adx:.2f}\n"
            f"Structure Summary: {structure_summary}\n"
            f"Provide a 2-3 sentence analysis of these indicators."
        )
        
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error fetching market commentary from LLM: {e}")
            return "Error retrieving commentary."

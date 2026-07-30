from typing import Protocol

from app.schemas import (
    AnalysisRequest,
    AnalysisResult,
    InspectionContext,
    VisionInspectionResult,
)


class VisionProvider(Protocol):
    async def inspect(self, image: bytes, context: InspectionContext) -> VisionInspectionResult: ...


class ReasoningProvider(Protocol):
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...


class ProviderError(RuntimeError):
    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message

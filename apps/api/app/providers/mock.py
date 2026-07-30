import hashlib

from app.enums import Disposition, RiskLevel
from app.schemas import (
    AnalysisRequest,
    AnalysisResult,
    BoundingBox,
    Defect,
    InspectionContext,
    VisionInspectionResult,
)


class MockVisionProvider:
    async def inspect(self, image: bytes, context: InspectionContext) -> VisionInspectionResult:
        bucket = hashlib.sha256(image).digest()[0] % 4
        if bucket == 0:
            return VisionInspectionResult(
                is_defective=False,
                overall_confidence=0.96,
                defects=[],
                summary=f"{context.product_code} 未发现可见缺陷",
            )

        risks = {1: RiskLevel.MEDIUM, 2: RiskLevel.HIGH, 3: RiskLevel.CRITICAL}
        risk = risks[bucket]
        defect_types = {1: "surface_scratch", 2: "edge_crack", 3: "structural_damage"}
        return VisionInspectionResult(
            is_defective=True,
            overall_confidence=0.82 + bucket * 0.04,
            defects=[
                Defect(
                    defect_type=defect_types[bucket],
                    confidence=0.78 + bucket * 0.05,
                    severity=risk,
                    description="Mock Provider 根据图像内容散列生成的可复现实例",
                    bounding_box=BoundingBox(x=0.22, y=0.18, width=0.31, height=0.24),
                )
            ],
            summary=f"发现 1 个 {risk.value} 风险缺陷",
        )


class MockReasoningProvider:
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        if not request.vision_result.is_defective:
            return AnalysisResult(
                risk_level=RiskLevel.LOW,
                probable_causes=["未发现异常"],
                recommended_actions=["放行产品并保留检测记录"],
                disposition=Disposition.RELEASE,
                requires_human_approval=False,
                rationale="视觉结果无缺陷且置信度满足放行规则。",
            )

        highest = max(
            (defect.severity for defect in request.vision_result.defects),
            key=lambda risk: list(RiskLevel).index(risk),
        )
        disposition = {
            RiskLevel.LOW: Disposition.RELEASE,
            RiskLevel.MEDIUM: Disposition.MANUAL_REVIEW,
            RiskLevel.HIGH: Disposition.REJECT,
            RiskLevel.CRITICAL: Disposition.STOP_LINE,
        }[highest]
        return AnalysisResult(
            risk_level=highest,
            probable_causes=["来料波动", "工艺参数漂移", "工装定位偏差"],
            recommended_actions=[
                "隔离当前产品",
                "复核同批次样本",
                "检查上游工艺参数",
            ],
            disposition=disposition,
            requires_human_approval=disposition == Disposition.STOP_LINE,
            rationale="根据缺陷严重度、置信度和质检规则生成处置建议。",
        )

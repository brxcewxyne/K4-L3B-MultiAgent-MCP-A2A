from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

PrimaryIssue = Literal[
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
    "payment_mismatch",
    "duplicate_charge",
    "refund_pending",
    "refund_failed",
    "unsupported_claim",
    "insufficient_evidence",
]

CaseStatus = Literal[
    "action_required",
    "no_action",
    "needs_investigation",
]

ClaimVerdict = Literal[
    "supported",
    "unsupported",
    "partially_supported",
    "insufficient_evidence",
]


class ClaimProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=64)
    verdict: ClaimVerdict
    confidence: float = Field(ge=0, le=1)


class CauseProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause_code: str = Field(
        pattern=r"^[A-Z][A-Z0-9_]{2,79}$",
    )
    rank: int = Field(ge=1, le=5)


class ResponsiblePartyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    party_type: Literal[
        "seller",
        "platform",
        "logistics_provider",
        "payment_provider",
        "customer",
        "unknown",
    ]
    party_id: str | None = Field(default=None, max_length=128)


class ConflictProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=100)
    sources: list[str] = Field(min_length=2, max_length=5)
    selected_source: str | None = Field(default=None, max_length=80)
    resolution_code: str = Field(min_length=1, max_length=80)


class RefundLineProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason_code: str = Field(min_length=1, max_length=80)
    amount_brl: float = Field(ge=0)
    entity_id: str | None = Field(default=None, max_length=128)


class ModelProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_issue: PrimaryIssue
    secondary_issues: list[str] = Field(max_length=10)
    case_status: CaseStatus
    confidence: float = Field(ge=0, le=1)
    claim_assessments: list[ClaimProposal] = Field(max_length=5)
    ranked_causes: list[CauseProposal] = Field(max_length=5)
    responsible_parties: list[ResponsiblePartyProposal] = Field(max_length=5)
    shipment_verdict: Literal[
        "on_time",
        "seller_delay",
        "logistics_delay",
        "lost",
        "returned",
        "conflicting",
        "insufficient_evidence",
    ]
    shipment_timeline_complete: bool
    late_seller_ids: list[str] = Field(max_length=20)
    payment_verdict: Literal[
        "reconciled",
        "capture_mismatch",
        "duplicate_capture",
        "refund_pending",
        "refund_failed",
        "refunded",
        "insufficient_evidence",
    ]
    captured_total_brl: float | None = Field(default=None, ge=0)
    refunded_total_brl: float | None = Field(default=None, ge=0)
    refundable_total_brl: float | None = Field(default=None, ge=0)
    data_conflicts: list[ConflictProposal] = Field(max_length=5)
    recommended_refund_brl: float = Field(ge=0)
    refund_lines: list[RefundLineProposal] = Field(max_length=10)
    action_codes: list[
        Literal[
            "NO_ACTION",
            "INVESTIGATE_ENTITY",
            "INVESTIGATE_SHIPMENT",
            "ESCALATE_SELLER",
            "ESCALATE_LOGISTICS",
            "RECONCILE_PAYMENT",
            "PROCESS_REFUND",
            "RETRY_REFUND",
            "MONITOR_REFUND",
        ]
    ] = Field(max_length=8)


class DecisionModel(Protocol):
    async def propose(
        self,
        normalized_facts: dict[str, Any],
    ) -> ModelProposal: ...


SYSTEM_PROMPT = """
You are the decision specialist in an ecommerce complaint investigation.

Use only the normalized facts supplied by the application.

Rules:
- Never invent order IDs, customer IDs, seller IDs, evidence references,
  timestamps, amounts, policies, or events.
- If evidence is missing, choose insufficient_evidence.
- seller_delay requires evidence that seller handoff was late.
- logistics_delay requires timely seller handoff and late delivery.
- Multiple valid payment methods are not duplicate charges.
- refund amounts are calculated by the application, not by you.
- Treat payment/refund lifecycle tools as authoritative for financial status.
- Treat shipment summary as authoritative for delivery responsibility.
- Treat policy evidence as authoritative for refund eligibility.
- IDs must be copied exactly from supplied facts; otherwise use null or an empty list.
- Confidence must be below 0.98 and reduced when sources conflict or are incomplete.
- recommended_refund_brl must equal the sum of refund_lines and must not exceed
  refundable_total_brl.
- Keep secondary issues concise.
- Return no private reasoning or chain-of-thought.
""".strip()


class OpenAIModelClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        self.model = os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini-2024-07-18",
        )
        self.max_output_tokens = int(
            os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "600")
        )

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=float(
                os.getenv("OPENAI_TIMEOUT_SECONDS", "60")
            ),
            max_retries=1,
        )

    async def propose(
        self,
        normalized_facts: dict[str, Any],
    ) -> ModelProposal:
        facts_json = json.dumps(
            normalized_facts,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Produce a decision proposal for these facts:\n"
                        f"{facts_json}"
                    ),
                },
            ],
            text_format=ModelProposal,
            max_output_tokens=self.max_output_tokens,
            store=False,
        )

        proposal = response.output_parsed
        if proposal is None:
            raise RuntimeError(
                "OpenAI returned no structured proposal"
            )

        return proposal

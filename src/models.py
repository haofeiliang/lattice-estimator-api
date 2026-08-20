"""Strict, versioned request/response types for the internal estimator API."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .constants import ADAPTER_SCHEMA_VERSION, DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS


class StrictModel(BaseModel):
    """Reject coercion and all unknown fields at every protocol boundary."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


CanonicalInteger: TypeAlias = Annotated[str, StringConstraints(pattern=r"^(0|[1-9][0-9]*)$")]
CanonicalSignedInteger: TypeAlias = Annotated[
    str, StringConstraints(pattern=r"^(0|-?[1-9][0-9]*)$")
]
CanonicalDecimal: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=r"^(?:0|-?(?:0\.[0-9]*[1-9]|[1-9][0-9]*(?:\.[0-9]*[1-9])?))$"),
]
PositiveU64: TypeAlias = Annotated[int, Field(strict=True, gt=0, le=2**64 - 1)]
NonNegativeU64: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]


class FiniteSampleCount(StrictModel):
    kind: Literal["finite"]
    count: PositiveU64


class UnlimitedSampleCount(StrictModel):
    kind: Literal["unlimited"]


SampleCount: TypeAlias = Annotated[
    FiniteSampleCount | UnlimitedSampleCount, Field(discriminator="kind")
]


class UniformBinary(StrictModel):
    kind: Literal["uniform_binary"]


class UniformTernary(StrictModel):
    kind: Literal["uniform_ternary"]


class SparseTernary(StrictModel):
    kind: Literal["sparse_ternary"]


class FixedWeightBinary(StrictModel):
    kind: Literal["fixed_weight_binary"]
    hamming_weight: NonNegativeU64


class FixedWeightTernary(StrictModel):
    kind: Literal["fixed_weight_ternary"]
    positive_weight: NonNegativeU64
    negative_weight: NonNegativeU64


class DiscreteGaussian(StrictModel):
    kind: Literal["discrete_gaussian"]
    standard_deviation: CanonicalDecimal

    @field_validator("standard_deviation")
    @classmethod
    def standard_deviation_is_positive(cls, value: str) -> str:
        if _decimal(value) <= 0:
            raise ValueError("standard_deviation must be positive")
        return value


class CenteredBinomial(StrictModel):
    kind: Literal["centered_binomial"]
    eta: PositiveU64


class UniformInteger(StrictModel):
    kind: Literal["uniform_integer"]
    lower: CanonicalSignedInteger
    upper: CanonicalSignedInteger

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> UniformInteger:
        if int(self.lower) > int(self.upper):
            raise ValueError("uniform lower bound must not exceed upper bound")
        return self


SecretDistribution: TypeAlias = Annotated[
    UniformBinary
    | UniformTernary
    | SparseTernary
    | FixedWeightBinary
    | FixedWeightTernary
    | DiscreteGaussian
    | CenteredBinomial
    | UniformInteger,
    Field(discriminator="kind"),
]
ErrorDistribution: TypeAlias = Annotated[
    DiscreteGaussian | CenteredBinomial | UniformInteger,
    Field(discriminator="kind"),
]


class LweProblem(StrictModel):
    kind: Literal["lwe"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    samples: SampleCount
    secret: SecretDistribution
    error: ErrorDistribution

    @model_validator(mode="after")
    def semantics_are_valid(self) -> LweProblem:
        _require_modulus(self.modulus)
        _require_secret_length(self.secret, self.dimension)
        return self


class NtruStructure(str, Enum):
    MATRIX = "matrix"
    CIRCULANT = "circulant"


WireNtruStructure: TypeAlias = Annotated[NtruStructure, Field(strict=False)]


class NtruProblem(StrictModel):
    kind: Literal["ntru"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    secret: SecretDistribution
    error: ErrorDistribution
    structure: WireNtruStructure

    @model_validator(mode="after")
    def semantics_are_valid(self) -> NtruProblem:
        _require_modulus(self.modulus)
        _require_secret_length(self.secret, self.dimension)
        return self


class SisNorm(str, Enum):
    L2 = "l2"
    L_INFINITY = "l_infinity"


WireSisNorm: TypeAlias = Annotated[SisNorm, Field(strict=False)]


class SisProblem(StrictModel):
    kind: Literal["sis"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    columns: PositiveU64
    length_bound: CanonicalDecimal
    norm: WireSisNorm

    @model_validator(mode="after")
    def semantics_are_valid(self) -> SisProblem:
        _require_modulus(self.modulus)
        if _decimal(self.length_bound) <= 0:
            raise ValueError("SIS length_bound must be positive")
        return self


EstimatorProblem: TypeAlias = Annotated[
    LweProblem | NtruProblem | SisProblem, Field(discriminator="kind")
]


class CostModel(str, Enum):
    BDGL16 = "BDGL16"
    LAA_MOS_POL14 = "LaaMosPol14"


class ShapeModel(str, Enum):
    GSA = "GSA"


class Attack(str, Enum):
    ARORA_GB = "arora_gb"
    BKW = "bkw"
    USVP = "usvp"
    BDD = "bdd"
    BDD_HYBRID = "bdd_hybrid"
    BDD_MITM_HYBRID = "bdd_mitm_hybrid"
    DUAL = "dual"
    DUAL_HYBRID = "dual_hybrid"
    DSD = "dsd"
    LATTICE = "lattice"


LWE_ATTACKS = (
    Attack.ARORA_GB,
    Attack.BKW,
    Attack.USVP,
    Attack.BDD,
    Attack.BDD_HYBRID,
    Attack.BDD_MITM_HYBRID,
    Attack.DUAL,
    Attack.DUAL_HYBRID,
)
LWE_FAST_ATTACKS = LWE_ATTACKS[2:]
LWE_SLOW_ATTACKS = LWE_ATTACKS[:2]
NTRU_ATTACKS = (
    Attack.USVP,
    Attack.DSD,
    Attack.BDD,
    Attack.BDD_HYBRID,
    Attack.BDD_MITM_HYBRID,
)
SIS_ATTACKS = (Attack.LATTICE,)
ATTACKS_BY_PROBLEM = {"lwe": LWE_ATTACKS, "ntru": NTRU_ATTACKS, "sis": SIS_ATTACKS}
EXACT_DISTRIBUTIONS = (
    "uniform_binary",
    "uniform_ternary",
    "sparse_ternary",
    "fixed_weight_binary",
    "fixed_weight_ternary",
    "discrete_gaussian",
    "centered_binomial",
    "uniform_integer",
)


WireCostModel: TypeAlias = Annotated[CostModel, Field(strict=False)]
WireShapeModel: TypeAlias = Annotated[ShapeModel, Field(strict=False)]
WireAttack: TypeAlias = Annotated[Attack, Field(strict=False)]


class EstimatorModels(StrictModel):
    cost_model: WireCostModel
    shape_model: WireShapeModel


class EstimateRequest(StrictModel):
    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    problem: EstimatorProblem
    models: EstimatorModels
    target_attacks: Annotated[list[WireAttack], Field(min_length=1)]
    timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=MAX_TIMEOUT_SECONDS)] = (
        DEFAULT_TIMEOUT_SECONDS
    )

    @model_validator(mode="after")
    def attacks_are_unique_and_supported(self) -> EstimateRequest:
        if len(set(self.target_attacks)) != len(self.target_attacks):
            raise ValueError("target_attacks must not contain duplicates")
        allowed = attacks_for_problem(self.problem)
        unsupported = [attack.value for attack in self.target_attacks if attack not in allowed]
        if unsupported:
            raise ValueError(
                f"attacks are not valid for {self.problem.kind}: {', '.join(unsupported)}"
            )
        return self


class IntegerMetric(StrictModel):
    kind: Literal["integer"]
    value: CanonicalSignedInteger


class DecimalMetric(StrictModel):
    kind: Literal["decimal"]
    value: CanonicalDecimal


class BooleanMetric(StrictModel):
    kind: Literal["boolean"]
    value: bool


class TextMetric(StrictModel):
    kind: Literal["text"]
    value: str


NormalizedMetric: TypeAlias = Annotated[
    IntegerMetric | DecimalMetric | BooleanMetric | TextMetric,
    Field(discriminator="kind"),
]


class ComputedOutcome(StrictModel):
    kind: Literal["computed"]
    security_bits: CanonicalDecimal
    metrics: dict[str, NormalizedMetric] = Field(default_factory=dict)


class UnsupportedOutcome(StrictModel):
    kind: Literal["unsupported"]
    code: str
    reason: str
    raw_result: JsonValue | None = None


class NoFiniteEstimateOutcome(StrictModel):
    kind: Literal["no_finite_estimate"]
    code: str
    reason: str
    raw_result: JsonValue | None = None


class FailedOutcome(StrictModel):
    kind: Literal["failed"]
    code: str
    message: str
    retryable: bool
    raw_result: JsonValue | None = None


WorkerOutcome: TypeAlias = Annotated[
    ComputedOutcome | NoFiniteEstimateOutcome | UnsupportedOutcome | FailedOutcome,
    Field(discriminator="kind"),
]


class AttackExecution(StrictModel):
    attack: Attack
    outcome: WorkerOutcome


class EstimatorProvenance(StrictModel):
    estimator_commit: str
    sage_version: str
    adapter_version: str
    adapter_schema_version: PositiveU64
    worker_image: str


class EstimateResponse(StrictModel):
    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    results: list[AttackExecution]
    duration_ms: NonNegativeU64
    provenance: EstimatorProvenance


class WorkerResponse(StrictModel):
    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    results: list[AttackExecution]
    duration_ms: NonNegativeU64


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    adapter_version: str


class SupportMatrixEntry(StrictModel):
    attacks: list[Attack]
    distributions: list[str]
    notes: list[str] = Field(default_factory=list)


class MetadataResponse(StrictModel):
    adapter_version: str
    adapter_schema_version: PositiveU64
    estimator_commit: str
    sage_version: str
    worker_image: str
    platform: Literal["linux/amd64"]
    support_matrix: dict[str, SupportMatrixEntry]
    adaptive_attacks: list[Attack]


class ErrorEnvelope(StrictModel):
    code: str
    message: str
    path: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


def attacks_for_problem(problem: EstimatorProblem) -> tuple[Attack, ...]:
    return ATTACKS_BY_PROBLEM[problem.kind]


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal") from error
    if not parsed.is_finite():
        raise ValueError("decimal must be finite")
    return parsed


def _require_modulus(value: str) -> None:
    if int(value) <= 1:
        raise ValueError("modulus must be greater than one")


def _require_secret_length(secret: SecretDistribution, logical_length: int) -> None:
    if isinstance(secret, FixedWeightBinary) and secret.hamming_weight > logical_length:
        raise ValueError("fixed binary weight exceeds logical secret length")
    if (
        isinstance(secret, FixedWeightTernary)
        and secret.positive_weight + secret.negative_weight > logical_length
    ):
        raise ValueError("fixed ternary weights exceed logical secret length")

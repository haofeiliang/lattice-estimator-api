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


# Integers and decimals cross the JSON boundary as canonical strings.  This
# avoids float rounding and language-specific limits for cryptographic values.
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
    """A caller-supplied finite number of available problem samples."""

    kind: Literal["finite"]
    count: PositiveU64


class UnlimitedSampleCount(StrictModel):
    """Request estimator semantics with no finite sample-count bound."""

    kind: Literal["unlimited"]


# Tagged unions use ``kind`` as their JSON discriminator throughout the protocol.
SampleCount: TypeAlias = Annotated[
    FiniteSampleCount | UnlimitedSampleCount, Field(discriminator="kind")
]


class UniformBinary(StrictModel):
    """Coefficient-wise uniform distribution over ``{0, 1}``."""

    kind: Literal["uniform_binary"]


class UniformTernary(StrictModel):
    """Coefficient-wise uniform distribution over ``{-1, 0, 1}``."""

    kind: Literal["uniform_ternary"]


class SparseTernary(StrictModel):
    """Probabilistic ternary distribution with probabilities 1/4, 1/2, 1/4."""

    kind: Literal["sparse_ternary"]


class FixedWeightBinary(StrictModel):
    """Binary secret with an exact Hamming weight."""

    kind: Literal["fixed_weight_binary"]
    hamming_weight: NonNegativeU64


class FixedWeightTernary(StrictModel):
    """Ternary secret with exact positive and negative coefficient counts."""

    kind: Literal["fixed_weight_ternary"]
    positive_weight: NonNegativeU64
    negative_weight: NonNegativeU64


class DiscreteGaussian(StrictModel):
    """Integer-valued centered discrete Gaussian distribution."""

    kind: Literal["discrete_gaussian"]
    standard_deviation: CanonicalDecimal

    @field_validator("standard_deviation")
    @classmethod
    def standard_deviation_is_positive(cls, value: str) -> str:
        """Reject zero and negative Gaussian standard deviations."""
        if _decimal(value) <= 0:
            raise ValueError("standard_deviation must be positive")
        return value


class CenteredBinomial(StrictModel):
    """Centered binomial distribution parameterized by ``eta`` trials per side."""

    kind: Literal["centered_binomial"]
    eta: PositiveU64


class UniformInteger(StrictModel):
    """Coefficient-wise uniform distribution on an inclusive integer interval."""

    kind: Literal["uniform_integer"]
    lower: CanonicalSignedInteger
    upper: CanonicalSignedInteger

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> UniformInteger:
        """Require a non-empty inclusive interval."""
        if int(self.lower) > int(self.upper):
            raise ValueError("uniform lower bound must not exceed upper bound")
        return self


# Secret distributions are broader than error distributions: fixed-weight and
# sparse variants model structured secrets but are not accepted as LWE errors.
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
    """Public LWE instance supplied to exact estimation or slow-attack preflight."""

    kind: Literal["lwe"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    samples: SampleCount
    secret: SecretDistribution
    error: ErrorDistribution

    @model_validator(mode="after")
    def semantics_are_valid(self) -> LweProblem:
        """Validate constraints that depend on multiple LWE fields."""
        _require_modulus(self.modulus)
        _require_secret_length(self.secret, self.dimension)
        return self


class NtruStructure(str, Enum):
    """Distinguish unstructured matrix NTRU from circulant polynomial NTRU."""

    MATRIX = "matrix"
    CIRCULANT = "circulant"


WireNtruStructure: TypeAlias = Annotated[NtruStructure, Field(strict=False)]


class NtruProblem(StrictModel):
    """Public NTRU instance, including its matrix or circulant structure."""

    kind: Literal["ntru"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    secret: SecretDistribution
    error: ErrorDistribution
    structure: WireNtruStructure

    @model_validator(mode="after")
    def semantics_are_valid(self) -> NtruProblem:
        """Validate modulus and fixed-weight secret constraints."""
        _require_modulus(self.modulus)
        _require_secret_length(self.secret, self.dimension)
        return self


class SisNorm(str, Enum):
    """Norm used to interpret an SIS solution-length bound."""

    L2 = "l2"
    L_INFINITY = "l_infinity"


WireSisNorm: TypeAlias = Annotated[SisNorm, Field(strict=False)]


class SisProblem(StrictModel):
    """Public SIS instance passed to the upstream SIS estimator."""

    kind: Literal["sis"]
    dimension: PositiveU64
    modulus: CanonicalInteger
    columns: PositiveU64
    length_bound: CanonicalDecimal
    norm: WireSisNorm

    @model_validator(mode="after")
    def semantics_are_valid(self) -> SisProblem:
        """Require a valid modulus and a positive solution-length bound."""
        _require_modulus(self.modulus)
        if _decimal(self.length_bound) <= 0:
            raise ValueError("SIS length_bound must be positive")
        return self


# A request carries exactly one problem family selected by its ``kind`` field.
EstimatorProblem: TypeAlias = Annotated[
    LweProblem | NtruProblem | SisProblem, Field(discriminator="kind")
]


class CostModel(str, Enum):
    """Supported lattice-reduction cost models exposed by the API."""

    BDGL16 = "BDGL16"
    LAA_MOS_POL14 = "LaaMosPol14"


class ShapeModel(str, Enum):
    """Supported Gram-Schmidt shape simulators exposed by the API."""

    GSA = "GSA"


class Attack(str, Enum):
    """Stable public attack identifiers independent of upstream display names."""

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


# These tuples define both validation and canonical response order.  Their order
# must stay stable because the parent verifies worker coverage positionally.
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
    """Cost and basis-shape assumptions shared by all requested attacks."""

    cost_model: WireCostModel
    shape_model: WireShapeModel


class EstimateRequest(StrictModel):
    """One exact-estimation request sent from HTTP to a Sage worker."""

    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    problem: EstimatorProblem
    models: EstimatorModels
    target_attacks: Annotated[list[WireAttack], Field(min_length=1)]
    timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=MAX_TIMEOUT_SECONDS)] = (
        DEFAULT_TIMEOUT_SECONDS
    )

    @model_validator(mode="after")
    def attacks_are_unique_and_supported(self) -> EstimateRequest:
        """Reject duplicates and attacks unsupported by the selected problem kind."""
        if len(set(self.target_attacks)) != len(self.target_attacks):
            raise ValueError("target_attacks must not contain duplicates")
        allowed = attacks_for_problem(self.problem)
        unsupported = [attack.value for attack in self.target_attacks if attack not in allowed]
        if unsupported:
            raise ValueError(
                f"attacks are not valid for {self.problem.kind}: {', '.join(unsupported)}"
            )
        return self


class PreflightRequest(EstimateRequest):
    """Request attack-specific cheap estimates without running the attacks."""

    operation: Literal["preflight"] = "preflight"
    required_security_bits: CanonicalDecimal
    requested_arora_gb_coarse_margin_bits: CanonicalDecimal
    requested_arora_gb_refined_margin_bits: CanonicalDecimal

    @field_validator("required_security_bits")
    @classmethod
    def required_security_is_positive(cls, value: str) -> str:
        """Require a positive target for threshold-based scheduling."""
        if _decimal(value) <= 0:
            raise ValueError("required_security_bits must be positive")
        return value

    @field_validator(
        "requested_arora_gb_coarse_margin_bits",
        "requested_arora_gb_refined_margin_bits",
    )
    @classmethod
    def requested_margin_is_non_negative(cls, value: str) -> str:
        """Allow callers to increase, but never invert, calibrated margins."""
        if _decimal(value) < 0:
            raise ValueError("requested Arora-GB margins must be non-negative")
        return value

    @model_validator(mode="after")
    def only_slow_lwe_attacks_are_allowed(self) -> PreflightRequest:
        """Restrict preflight to the two supported slow LWE attacks."""
        if not isinstance(self.problem, LweProblem):
            raise ValueError("preflight is only available for LWE")
        unsupported = [
            attack.value for attack in self.target_attacks if attack not in LWE_SLOW_ATTACKS
        ]
        if unsupported:
            raise ValueError(f"preflight only supports arora_gb and bkw: {', '.join(unsupported)}")
        return self


class IntegerMetric(StrictModel):
    """Normalized integer-valued diagnostic safe for JSON transport."""

    kind: Literal["integer"]
    value: CanonicalSignedInteger


class DecimalMetric(StrictModel):
    """Normalized finite decimal diagnostic encoded canonically as text."""

    kind: Literal["decimal"]
    value: CanonicalDecimal


class BooleanMetric(StrictModel):
    """Normalized Boolean diagnostic emitted by an attack adapter."""

    kind: Literal["boolean"]
    value: bool


class TextMetric(StrictModel):
    """Normalized textual diagnostic emitted by an attack adapter."""

    kind: Literal["text"]
    value: str


# Raw Sage objects never cross the process boundary; only these scalar metric
# variants are retained for audit and UI diagnostics.
NormalizedMetric: TypeAlias = Annotated[
    IntegerMetric | DecimalMetric | BooleanMetric | TextMetric,
    Field(discriminator="kind"),
]


class ComputedOutcome(StrictModel):
    """A finite attack cost that may contribute to the final security minimum."""

    kind: Literal["computed"]
    security_bits: CanonicalDecimal
    metrics: dict[str, NormalizedMetric] = Field(default_factory=dict)


class UnsupportedOutcome(StrictModel):
    """The requested attack is not implemented for the supplied parameter domain."""

    kind: Literal["unsupported"]
    code: str
    reason: str
    raw_result: JsonValue | None = None


class NoFiniteEstimateOutcome(StrictModel):
    """The exact estimator completed but found no finite positive attack cost."""

    kind: Literal["no_finite_estimate"]
    code: str
    reason: str
    raw_result: JsonValue | None = None


class PreflightUnknownOutcome(StrictModel):
    """A bounded preflight search could not make a scheduling decision."""

    kind: Literal["preflight_unknown"]
    code: str
    reason: str
    raw_result: JsonValue | None = None


class ThresholdScreenOutcome(StrictModel):
    """A target-aware Arora-GB scheduling decision, never a security result."""

    kind: Literal["threshold_screen"]
    decision: Literal["above_threshold", "needs_exact"]
    precision_tier: Literal["coarse", "refined"]
    required_security_bits: CanonicalDecimal
    requested_margin_bits: CanonicalDecimal
    calibrated_margin_floor_bits: CanonicalDecimal
    effective_margin_bits: CanonicalDecimal
    decision_threshold_bits: CanonicalDecimal
    reason: str
    metrics: dict[str, NormalizedMetric] = Field(default_factory=dict)


class FailedOutcome(StrictModel):
    """An attack or adapter failed before producing a usable conclusion."""

    kind: Literal["failed"]
    code: str
    message: str
    retryable: bool
    raw_result: JsonValue | None = None


# Every attack has exactly one tagged terminal/preflight outcome.  In particular,
# a threshold screen is scheduling evidence rather than a computed security bit.
WorkerOutcome: TypeAlias = Annotated[
    ComputedOutcome
    | NoFiniteEstimateOutcome
    | PreflightUnknownOutcome
    | ThresholdScreenOutcome
    | UnsupportedOutcome
    | FailedOutcome,
    Field(discriminator="kind"),
]


class AttackExecution(StrictModel):
    """One attack outcome plus timing ownership for an exact or preflight request."""

    attack: Attack
    outcome: WorkerOutcome
    duration_ms: NonNegativeU64 = 0
    duration_scope: Literal["attack", "request_group"] = "attack"
    shared_attacks: list[Attack] = Field(default_factory=list)


class EstimatorProvenance(StrictModel):
    """Pinned implementation versions needed to audit an estimate."""

    estimator_commit: str
    sage_version: str
    adapter_version: str
    adapter_schema_version: PositiveU64
    worker_image: str


class EstimateResponse(StrictModel):
    """Public HTTP response containing attack results and build provenance."""

    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    results: list[AttackExecution]
    duration_ms: NonNegativeU64
    provenance: EstimatorProvenance


class WorkerResponse(StrictModel):
    """Internal child-process response before HTTP provenance is attached."""

    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    results: list[AttackExecution]
    duration_ms: NonNegativeU64


class HealthResponse(StrictModel):
    """Minimal response used by container and orchestration health checks."""

    status: Literal["ok"] = "ok"
    adapter_version: str


class SupportMatrixEntry(StrictModel):
    """Supported attacks and distributions for one public problem kind."""

    attacks: list[Attack]
    distributions: list[str]
    notes: list[str] = Field(default_factory=list)


class MetadataResponse(StrictModel):
    """Service versions and capabilities returned by ``GET /v1/metadata``."""

    adapter_version: str
    adapter_schema_version: PositiveU64
    estimator_commit: str
    sage_version: str
    worker_image: str
    platform: Literal["linux/amd64"]
    support_matrix: dict[str, SupportMatrixEntry]
    adaptive_attacks: list[Attack]


class ErrorEnvelope(StrictModel):
    """Stable JSON shape used for validation, worker, and server errors."""

    code: str
    message: str
    path: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


def attacks_for_problem(problem: EstimatorProblem) -> tuple[Attack, ...]:
    """Return attacks allowed by the strict model for a problem variant."""
    return ATTACKS_BY_PROBLEM[problem.kind]


def _decimal(value: str) -> Decimal:
    """Parse a canonical decimal string and reject infinities and NaN."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal") from error
    if not parsed.is_finite():
        raise ValueError("decimal must be finite")
    return parsed


def _require_modulus(value: str) -> None:
    """Enforce the common cryptographic modulus lower bound."""
    if int(value) <= 1:
        raise ValueError("modulus must be greater than one")


def _require_secret_length(secret: SecretDistribution, logical_length: int) -> None:
    """Ensure fixed secret weights fit within the logical secret length."""
    if isinstance(secret, FixedWeightBinary) and secret.hamming_weight > logical_length:
        raise ValueError("fixed binary weight exceeds logical secret length")
    if (
        isinstance(secret, FixedWeightTernary)
        and secret.positive_weight + secret.negative_weight > logical_length
    ):
        raise ValueError("fixed ternary weights exceed logical secret length")

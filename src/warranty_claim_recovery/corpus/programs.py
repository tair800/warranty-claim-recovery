"""The warranty programmes, the parts they cover, and the coverage table a claim is judged against.

A **warranty programme** is a manufacturer plus a policy version, and it is the unit ADR-001 §7
splits the hold-out on. Everything that decides an outcome hangs off it: the clause text, the
warranty length, the correction window, the labour-rate cap, the deductible and the per-claim cap.
That is why the split cannot be taken at claim level — forty claims under one programme are
adjudicated against one policy, so a claim-level split would measure how well the system memorised a
policy it had already been tuned on.

Three properties of this module are deliberate.

**Six manufacturers, three policy versions each.** Eighteen programmes, not twelve. The contract
floor is twelve; the corpus clears it with margin because the hold-out rule takes roughly thirty per
cent of programmes and a hold-out of three or four programmes makes every rate measured over it a
rate decided by a handful of claims. The versions are the three oldest each manufacturer has in this
corpus, chosen by a stated rule rather than picked one at a time — a programme list assembled
identifier by identifier could be assembled until the split flattered a number, and this one cannot.

**The same part number can be covered under one policy version and excluded under another.** Parts
belong to the manufacturer; coverage belongs to the programme. Without that, filtering retrieval by
policy version would be decoration: every version would say the same thing and a system that ignored
the version would score identically to one that respected it.

**Coverage is a table, not a rule in code.** `PartCoverage` is what the eligibility check consumes,
and the corpus supplies one row per (programme, part) so a reviewer can read why a claim failed
instead of reconstructing a predicate. A part with no row is a claim the generator refuses to emit:
an eligibility check with nothing to check silently passes, and a silently passing eligibility check
is how an unsupported resubmission leaves the system.

Rejected: generating the manufacturers' names from the seed as well. Readable identifiers are worth
more than one more randomised axis, and an artifact whose programme column reads `prog-07` forces
every reader to hold a lookup table in their head while they check a claim.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final, NamedTuple

from warranty_claim_recovery.corpus.rng import stream
from warranty_claim_recovery.domain import WarrantyProgram
from warranty_claim_recovery.money import Currency, Money

__all__ = [
    "COVERED_BY_FAMILY",
    "FAMILY_CODES",
    "MANUFACTURERS",
    "POLICY_VERSIONS",
    "SERIAL_RANGE_DECLARED",
    "VARIANTS",
    "CoverageSpec",
    "Manufacturer",
    "PartFamily",
    "ProgramSpec",
    "build_programs",
    "part_number",
]


class Manufacturer(NamedTuple):
    """One manufacturer, its identifier stem, its trading currency and its part prefix."""

    slug: str
    name: str
    currency: Currency
    part_prefix: str


#: Six manufacturers across the three currencies `money.Currency` admits, two each. Two per
#: currency rather than one, so a currency-specific defect cannot hide behind a single programme's
#: arithmetic.
MANUFACTURERS: Final[tuple[Manufacturer, ...]] = (
    Manufacturer("kestrel-hydraulics", "Kestrel Hydraulics", Currency.GBP, "KH"),
    Manufacturer("northfield-drives", "Northfield Drives", Currency.GBP, "ND"),
    Manufacturer("meridian-pneumatics", "Meridian Pneumatics", Currency.EUR, "MP"),
    Manufacturer("aldersgate-motors", "Aldersgate Motors", Currency.EUR, "AM"),
    Manufacturer("brightwater-controls", "Brightwater Controls", Currency.USD, "BC"),
    Manufacturer("calder-thermal", "Calder Thermal", Currency.USD, "CT"),
)

#: Three policy versions per manufacturer. The list is the same shape for every manufacturer and the
#: minor numbers differ, so no two programme identifiers collide and none was chosen individually.
POLICY_VERSIONS: Final[dict[str, tuple[str, ...]]] = {
    "kestrel-hydraulics": ("2019.1", "2020.2", "2021.3"),
    "northfield-drives": ("2019.2", "2020.4", "2021.1"),
    "meridian-pneumatics": ("2019.4", "2020.1", "2021.2"),
    "aldersgate-motors": ("2019.1", "2020.3", "2021.4"),
    "brightwater-controls": ("2019.3", "2020.2", "2021.1"),
    "calder-thermal": ("2019.2", "2020.1", "2021.3"),
}

#: Four part families per manufacturer, and the code is the same across manufacturers so that a
#: reader can tell at a glance that `KH-PMP-…` and `ND-PMP-…` are the same kind of component under
#: two different warranties. The numeric block differs per manufacturer and is drawn from the seed.
FAMILY_CODES: Final[tuple[tuple[str, str], ...]] = (
    ("PMP", "hydraulic pump"),
    ("VLV", "control valve"),
    ("SNS", "temperature sensor assembly"),
    ("DRV", "drive unit"),
)

#: Four variants per family. The variant is where "similar part family, wrong exact variant" lives:
#: `…-4120-A` and `…-4120-C` differ by one character and by whether the claim is recoverable at all.
VARIANTS: Final[tuple[str, ...]] = ("A", "B", "C", "D")

#: Which variants each family covers, by family position. Fixed rather than drawn, because each row
#: exists to make one adversarial construction possible and a random table would sometimes make it
#: impossible:
#:
#:   position 0  A and B covered, C and D not   -> "similar part family, wrong exact variant"
#:   position 1  everything covered             -> the ordinary, recoverable case
#:   position 2  nothing covered                -> "exclusion clause"
#:   position 3  B not covered, the rest are    -> "supplier changed after manufacture date"
COVERED_BY_FAMILY: Final[tuple[tuple[bool, ...], ...]] = (
    (True, True, False, False),
    (True, True, True, True),
    (False, False, False, False),
    (True, False, True, True),
)

#: Whether the programme declares a serial range for that (family, variant). A row with no declared
#: range covers every serial, and the distinction is load-bearing: a claim whose serial is unknown
#: can still be adjudicated under a part with no declared range, and cannot be adjudicated under one
#: with a range. The corpus routes its "missing serial evidence" claims at the former on purpose, so
#: that the outcome turns on the missing evidence rather than on an unstated reading of what an
#: unknown serial means against a declared range.
SERIAL_RANGE_DECLARED: Final[tuple[tuple[bool, ...], ...]] = (
    (True, True, True, True),
    (True, True, False, False),
    (False, False, False, False),
    (True, False, True, True),
)

#: The note carried on an uncovered row, by family position. It is the sentence the console shows a
#: person, and it names the clause the exclusion comes from rather than asserting "not covered".
_EXCLUSION_NOTES: Final[dict[int, str]] = {
    0: "variant not listed in the covered-parts schedule for this policy version",
    2: "excluded as a serviceable consumable under the covered-parts and exclusions clause",
    3: "component supplier changed after the equipment manufacture date; cover did not carry over",
}

_COVERED_NOTE: Final = "listed in the covered-parts schedule for this policy version"

#: Deductible options, in major units. Zero is in the list deliberately. A corpus where every
#: programme charges a deductible cannot distinguish "the deductible reduced the recovery" from "the
#: recovery was reduced", and the ground-truth rule's definition of a full recovery would then be
#: untestable because no claim could ever meet it.
_DEDUCTIBLE_CHOICES: Final[tuple[int, ...]] = (0, 0, 2500, 5000, 7500, 10000, 15000)


class PartFamily(NamedTuple):
    """A family of parts that differ only by variant letter."""

    family_id: str
    label: str
    position: int


class CoverageSpec(NamedTuple):
    """One row of the programme's covered-parts schedule.

    `covered`, `serial_first` and `serial_last` are exactly the fields `eligibility.PartCoverage`
    consumes. `note` and `program_id` are corpus metadata and are carried beside that tuple in the
    generated file rather than inside it, so a loader can build the tuple by field name without
    tripping the `extra="forbid"` that guards every model in this system.
    """

    program_id: str
    part_number: str
    family_id: str
    variant: str
    covered: bool
    serial_first: int | None
    serial_last: int | None
    note: str


class ProgramSpec(NamedTuple):
    """A programme with everything the claim builder needs to construct claims against it."""

    program: WarrantyProgram
    manufacturer: Manufacturer
    families: tuple[PartFamily, ...]
    coverage: tuple[CoverageSpec, ...]

    def coverage_for(self, number: str) -> CoverageSpec:
        for row in self.coverage:
            if row.part_number == number:
                return row
        raise KeyError(
            f"{self.program.program_id} has no coverage row for {number}. A claim whose part is "
            f"absent from the schedule is a claim whose eligibility check has nothing to check, "
            f"and "
            f"an eligibility check with nothing to check passes silently."
        )


def part_number(prefix: str, family_id: str, variant: str) -> str:
    """The claim's part number.

    `prefix` is already inside `family_id`, and is taken again so that a call site reads as
    the assertion it is: this part belongs to this manufacturer. The check below is the
    assertion actually being made.
    """
    if not family_id.startswith(prefix):
        raise ValueError(f"{family_id} does not belong to the {prefix} prefix")
    return f"{family_id}-{variant}"


def _families(manufacturer: Manufacturer) -> tuple[PartFamily, ...]:
    draw = stream("part-families", manufacturer.slug)
    # One numeric block per manufacturer, drawn once and reused across its families with a fixed
    # offset, so the four families of a manufacturer read as a range rather than as four unrelated
    # numbers. The offset is deterministic and the block is keyed by the manufacturer, so adding a
    # seventh manufacturer moves nothing about the first six.
    block = draw.randrange(1000, 9000)
    return tuple(
        PartFamily(
            family_id=f"{manufacturer.part_prefix}-{code}-{block + position * 120}",
            label=label,
            position=position,
        )
        for position, (code, label) in enumerate(FAMILY_CODES)
    )


def _coverage(program_id: str, families: tuple[PartFamily, ...]) -> tuple[CoverageSpec, ...]:
    rows: list[CoverageSpec] = []
    for family in families:
        for index, variant in enumerate(VARIANTS):
            number = f"{family.family_id}-{variant}"
            covered = COVERED_BY_FAMILY[family.position][index]
            declared = SERIAL_RANGE_DECLARED[family.position][index]
            first: int | None = None
            last: int | None = None
            if declared:
                # Keyed by the programme *and* the part, so the same part carries a different
                # declared range under a different policy version. A range shared across versions
                # would make the policy-version filter cosmetic.
                draw = stream("serial-range", program_id, number)
                first = 1_000_000 + draw.randrange(0, 80) * 100_000
                last = first + 49_999
            note = _COVERED_NOTE if covered else _EXCLUSION_NOTES[family.position]
            rows.append(
                CoverageSpec(
                    program_id=program_id,
                    part_number=number,
                    family_id=family.family_id,
                    variant=variant,
                    covered=covered,
                    serial_first=first,
                    serial_last=last,
                    note=note,
                )
            )
    return tuple(rows)


def _money(minor_units: int, currency: Currency) -> Money:
    """Build an amount from whole minor units, so no decimal string is ever parsed from a float.

    `Decimal(n).scaleb(-2)` is exact. `Decimal(n / 100)` would not be: the division happens in
    binary floating point before `Decimal` ever sees it, and the value would arrive already wrong
    while looking exact — the precise laundering `money.parse_money` refuses at the boundary.
    """
    return Money(Decimal(minor_units).scaleb(-2), currency)


def _build_program(manufacturer: Manufacturer, version: str) -> WarrantyProgram:
    program_id = f"{manufacturer.slug}-{version}"
    draw = stream("program-terms", program_id)
    # Every term is drawn in whole minor units and converted once. Ranges are chosen so that the
    # per-claim cap always exceeds the largest labour bill a claim in this corpus can carry, which
    # is what lets the claim builder decide whether the cap bites by construction rather than by
    # luck.
    labour_cap = _money(draw.randrange(4200, 9600, 50), manufacturer.currency)
    deductible = _money(draw.choice(_DEDUCTIBLE_CHOICES), manufacturer.currency)
    claim_cap = _money(draw.randrange(180_000, 450_000, 2_500), manufacturer.currency)
    return WarrantyProgram(
        program_id=program_id,
        manufacturer=manufacturer.name,
        policy_version=version,
        currency=manufacturer.currency,
        correction_window_days=draw.choice((21, 30, 45, 60, 90)),
        warranty_months=draw.choice((12, 18, 24, 36, 48, 60)),
        labour_rate_cap_per_hour=labour_cap,
        deductible=deductible,
        claim_cap=claim_cap,
    )


def build_programs() -> tuple[ProgramSpec, ...]:
    """Every programme in the corpus, in a fixed order.

    The order is manufacturer order then version order — both written out above — and never the
    iteration order of a set or a dictionary built at runtime. An artifact whose programme order
    depends on hash ordering is an artifact whose bytes differ between two runs for no reason a
    reader could ever diagnose.
    """
    specs: list[ProgramSpec] = []
    for manufacturer in MANUFACTURERS:
        families = _families(manufacturer)
        for version in POLICY_VERSIONS[manufacturer.slug]:
            program = _build_program(manufacturer, version)
            specs.append(
                ProgramSpec(
                    program=program,
                    manufacturer=manufacturer,
                    families=families,
                    coverage=_coverage(program.program_id, families),
                )
            )
    return tuple(specs)

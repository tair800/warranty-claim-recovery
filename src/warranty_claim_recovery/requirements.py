"""What each rejection code demands, as a table rather than as a judgement.

A manufacturer does not reject a claim and then leave the distributor to guess. It returns a code,
and behind that code sits a fixed list of things the corrected claim has to show. The failure this
module exists to prevent is the one the whole project is named after: a correction goes back still
missing the evidence it was rejected for the first time, the manufacturer rejects it again, and by
the time anybody reads the second rejection the claim window has shut and the money is written off.

Three properties are load-bearing here, and each of them is a decision that could have gone the
other way.

**The matrix is a total function over `RejectionCode`.** Every one of the seven codes maps to at
least one requirement, and a code the table does not know raises. The tempting alternative — return
an empty tuple for an unrecognised code — is the dangerous one, because an empty requirement list
does not mean "unknown", it means "nothing is required". `gate.decide` reads a fully satisfied
requirement list as permission to authorise money, so a code that fell through to an empty list
would produce a claim that is recoverable precisely because the system has no idea what it was
rejected for. `assert_matrix_is_total` runs at import, so adding an eighth rejection code without
its requirements breaks the package immediately rather than at the first claim that carries it.

**The requirements belong to the code, not to the handler.** The rejected alternative was a
free-form checklist written per case by whoever picked it up. It was rejected because two handlers
then ask for different evidence for the same rejection code, and the recovery rate becomes a
property of who was on shift rather than of the policy. A table is also the only version of this
that a manufacturer's adjudicator could be shown and would recognise.

**No model derives them.** ADR-001 §2 puts requirement determination inside the deterministic core
for the same reason it puts the money there: a requirement invented from the wording of a rejection
letter is a requirement that sends a technician to look for a document that does not exist, and the
case then waits days for evidence nobody could ever have produced.

`evidence_key` is the name under which the case's evidence bundle carries the artefact that would
satisfy the requirement. Keys are deliberately **not** unique across codes: one commissioning
certificate can satisfy a requirement raised by `MISSING_INSTALL_PROOF` and another raised by
`OUTSIDE_WARRANTY_PERIOD`, and forcing distinct keys would mean holding the same document twice
under two names, which is how the two copies come to disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, NamedTuple

from warranty_claim_recovery.domain import RejectionCode

__all__ = [
    "MAX_REQUIREMENTS_PER_CODE",
    "MIN_REQUIREMENTS_PER_CODE",
    "REQUIREMENT_MATRIX",
    "RequirementSpec",
    "UnknownRejectionCodeError",
    "assert_matrix_is_total",
    "required_for",
]


class RequirementSpec(NamedTuple):
    """One thing a rejection code demands, before any evidence has been looked at.

    A `NamedTuple` rather than a Pydantic model because these are compile-time constants in a
    literal table: there is no boundary here for validation to sit on, and paying a validation pass
    to re-check a tuple this module wrote itself would be ceremony. `Requirement` in `domain.py` is
    the validated form, built once the evidence has been weighed, and that one is a frozen model
    because it arrives from outside.
    """

    #: Stable across policy versions and across manufacturers. It appears in audit records and in
    #: the console, so renaming one is a breaking change to the evidence trail, not a tidy-up.
    requirement_id: str
    #: Written for the technician who has to go and find the thing. `Requirement.description`
    #: enforces a minimum length; a description of "serial" tells nobody what to fetch.
    description: str
    #: Where the case's evidence bundle carries the artefact that would satisfy this. Shared
    #: between codes on purpose; see the module docstring.
    evidence_key: str


#: A code with nothing behind it would authorise a resubmission that proves nothing, so the floor
#: is one. The ceiling is three because a rejection that demands four separate artefacts is two
#: rejections that were merged, and splitting them is the honest fix rather than a longer list.
MIN_REQUIREMENTS_PER_CODE: Final = 1
MAX_REQUIREMENTS_PER_CODE: Final = 3


class UnknownRejectionCodeError(LookupError):
    """A rejection code arrived that the requirement matrix has no entry for.

    A distinct type because the caller has something specific to do about it, and because the
    alternative — a bare `KeyError` from a dictionary lookup — reads, in a traceback three frames
    away, as though a case identifier were missing rather than as though this system does not know
    what the manufacturer asked for.
    """


REQUIREMENT_MATRIX: Final[dict[RejectionCode, tuple[RequirementSpec, ...]]] = {
    RejectionCode.MISSING_SERIAL: (
        RequirementSpec(
            requirement_id="REQ-SERIAL-STAMPED",
            description=(
                "The serial number stamped on the failed part, transcribed from the part itself "
                "rather than from the machine's build plate."
            ),
            evidence_key="part_serial_number",
        ),
        RequirementSpec(
            requirement_id="REQ-SERIAL-IN-COVERED-RANGE",
            description=(
                "Confirmation that the stamped serial falls inside the build range the warranty "
                "programme covers for this part number."
            ),
            evidence_key="serial_coverage_range",
        ),
    ),
    RejectionCode.MISSING_INSTALL_PROOF: (
        RequirementSpec(
            requirement_id="REQ-INSTALL-CERTIFICATE",
            description=(
                "The commissioning certificate showing the date the machine was put into service, "
                "which is the date the warranty period runs from."
            ),
            evidence_key="installation_certificate",
        ),
        RequirementSpec(
            requirement_id="REQ-INSTALL-ENGINEER",
            description=(
                "The identity of the engineer who commissioned the machine, so the manufacturer "
                "can confirm the installation was carried out by an approved party."
            ),
            evidence_key="installer_identity",
        ),
    ),
    RejectionCode.WRONG_FAILURE_CODE: (
        RequirementSpec(
            requirement_id="REQ-FAILURE-CODE-CORRECTED",
            description=(
                "The manufacturer failure code that matches the symptom actually recorded, drawn "
                "from the code list published with this policy version."
            ),
            evidence_key="failure_code",
        ),
        RequirementSpec(
            requirement_id="REQ-FAILURE-NARRATIVE",
            description=(
                "The technician's account of the symptom and the diagnosis, in enough detail that "
                "the corrected failure code can be checked against it."
            ),
            evidence_key="technician_report",
        ),
    ),
    RejectionCode.PART_NOT_COVERED: (
        RequirementSpec(
            requirement_id="REQ-PART-IN-SCHEDULE",
            description=(
                "The entry in the covered-parts schedule for this policy version that lists the "
                "claimed part number."
            ),
            evidence_key="covered_parts_schedule",
        ),
        RequirementSpec(
            requirement_id="REQ-PART-CAUSATION",
            description=(
                "Evidence that the claimed part caused the failure rather than being collateral "
                "damage from a part the policy excludes."
            ),
            evidence_key="failure_causation",
        ),
    ),
    RejectionCode.OUTSIDE_WARRANTY_PERIOD: (
        RequirementSpec(
            requirement_id="REQ-IN-SERVICE-DATE",
            description=(
                "The in-service date from the commissioning record, which is what the warranty "
                "period is measured from when the invoice date and the despatch date disagree."
            ),
            evidence_key="in_service_record",
        ),
        RequirementSpec(
            requirement_id="REQ-WARRANTY-PERIOD-CLAUSE",
            description=(
                "The policy clause stating the length of the warranty period for this product "
                "family, in the policy version in force on the in-service date."
            ),
            evidence_key="warranty_period_clause",
        ),
    ),
    RejectionCode.DUPLICATE_CLAIM: (
        RequirementSpec(
            requirement_id="REQ-PRIOR-RECOVERY-STATEMENT",
            description=(
                "A statement of what has already been recovered against this machine, so that the "
                "amount now claimed can be shown to be the remainder rather than a repeat."
            ),
            evidence_key="prior_recovery_ledger",
        ),
        RequirementSpec(
            requirement_id="REQ-DISTINCT-RECOVERY-IDENTITY",
            description=(
                "Evidence that this claim, part and serial together differ from the recovery the "
                "manufacturer has already settled."
            ),
            evidence_key="recovery_identity",
        ),
    ),
    RejectionCode.LABOUR_RATE_EXCEEDED: (
        RequirementSpec(
            requirement_id="REQ-LABOUR-RATE-SCHEDULE",
            description=(
                "The labour rate charged on the repair invoice, set against the rate cap this "
                "warranty programme publishes."
            ),
            evidence_key="labour_rate_schedule",
        ),
        RequirementSpec(
            requirement_id="REQ-LABOUR-HOURS-ALLOWANCE",
            description=(
                "The hours charged, set against the manufacturer's published repair time "
                "allowance for this operation."
            ),
            evidence_key="repair_time_allowance",
        ),
    ),
}


def assert_matrix_is_total(
    matrix: Mapping[RejectionCode, tuple[RequirementSpec, ...]] | None = None,
) -> None:
    """Refuse a matrix that could let a claim through with nothing required of it.

    Three things are checked, and each corresponds to a way the table has gone wrong in practice
    rather than in theory.

    First, **totality**. A rejection code with no entry is the failure described at length in the
    module docstring: the gate reads "no unsatisfied requirements" as permission.

    Second, **the bounds**. An empty tuple is the same failure wearing a different hat — the key
    exists, so a `in matrix` check passes, and the list behind it still requires nothing.

    Third, **globally unique requirement identifiers**. Two codes that both emit
    `REQ-SERIAL-STAMPED` would produce audit records in which a satisfied requirement cannot be
    traced back to the rejection that raised it, and kill condition H is graded from exactly those
    records.

    Takes the matrix as an argument, defaulting to the shipped one, so that the test can hand it a
    broken table and observe the refusal. A guard that can only ever be run against the correct
    input has never been shown to reject anything.
    """
    subject = REQUIREMENT_MATRIX if matrix is None else matrix

    absent = [code.value for code in RejectionCode if code not in subject]
    if absent:
        raise UnknownRejectionCodeError(
            f"the requirement matrix has no entry for {absent}. A rejection code with no "
            f"requirements is read downstream as a claim with nothing outstanding, which is how "
            f"an unevidenced resubmission is authorised."
        )

    # Iterated over the enum rather than over the mapping. Both are ordered in CPython, but only
    # the enum's order is declared: an error message that names a different one of two offending
    # codes depending on how the table was built is a message nobody can act on twice.
    for code in RejectionCode:
        specs = subject[code]
        if not MIN_REQUIREMENTS_PER_CODE <= len(specs) <= MAX_REQUIREMENTS_PER_CODE:
            raise ValueError(
                f"{code.value} declares {len(specs)} requirement(s); the table permits "
                f"{MIN_REQUIREMENTS_PER_CODE} to {MAX_REQUIREMENTS_PER_CODE}"
            )

    seen: dict[str, RejectionCode] = {}
    for code in RejectionCode:
        for spec in subject[code]:
            first = seen.setdefault(spec.requirement_id, code)
            if first is not code:
                raise ValueError(
                    f"{spec.requirement_id} is declared by both {first.value} and {code.value}; a "
                    f"satisfied requirement would then be untraceable to the rejection that "
                    f"raised it"
                )


def required_for(code: RejectionCode) -> tuple[RequirementSpec, ...]:
    """Everything this rejection code demands, in a fixed order.

    The order is the table's, and the table's order is the order a technician should work in: the
    artefact that is hardest to obtain is never first, because a case that stalls on its first
    requirement never reveals the other two, and the second round of chasing costs another day
    against a window that is already running.
    """
    try:
        return REQUIREMENT_MATRIX[code]
    except KeyError as error:
        raise UnknownRejectionCodeError(
            f"no requirements are declared for rejection code {code!r}. This system will not "
            f"treat an unknown rejection as a claim with nothing outstanding."
        ) from error


# Run at import. A table that is wrong is wrong for every claim that follows, so the cheapest place
# to find out is the moment the package is loaded — before a corpus has been generated, before a
# worker has leased a case, and while the traceback still points at this file.
assert_matrix_is_total()

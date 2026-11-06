from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaxpayerInput:
    """Structured input both regimes' calculators consume."""
    annual_salary: float
    basic_salary: Optional[float] = None
    hra_received: Optional[float] = None
    rent_paid: Optional[float] = None
    city_is_metro: Optional[bool] = None
    section_80c: float = 0.0
    section_80ccd1b: float = 0.0
    section_80d_self: float = 0.0
    section_80d_parents: float = 0.0
    parents_are_senior_citizens: bool = False
    home_loan_interest_self_occupied: float = 0.0
    age: Optional[int] = None
    # extend as needed; keep every field traceable to a clause in clause_refs.py


@dataclass
class TaxResult:
    regime: str                     # "old" or "new"
    gross_total_income: float
    total_deductions: float
    taxable_income: float
    tax_before_cess: float
    surcharge: float
    cess: float
    rebate_87a: float
    final_tax_payable: float
    clause_trace: list[str] = field(default_factory=list)  # e.g. ["Sec 80C", "Sec 87A"]

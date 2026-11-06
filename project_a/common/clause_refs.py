"""
Statutory Clause Citations and Audit Trail Registry (Project A).

Provides machine-readable constants and helpers to trace every rupee calculated
back to its governing section of the Income-tax Act, 1961 or relevant Finance Act.
"""

from typing import Iterable


# Section References
SEC_16_IA_OLD = "Sec 16(ia) - Standard Deduction (Old Regime ₹50,000)"
SEC_16_IA_NEW = "Sec 16(ia) - Standard Deduction (New Regime ₹75,000)"

SEC_10_13A_HRA = "Sec 10(13A) read with Rule 2A - HRA Exemption"
SEC_80C = "Sec 80C - Qualifying Investments (Cap ₹1,50,000)"
SEC_80CCD_1B = "Sec 80CCD(1B) - Additional NPS Contribution (Cap ₹50,000)"
SEC_80D_SELF = "Sec 80D - Health Insurance (Self/Family)"
SEC_80D_PARENTS = "Sec 80D - Health Insurance (Parents)"
SEC_24B = "Sec 24(b) - Interest on Borrowed Capital (Self-Occupied House ₹2,00,000)"

SLAB_OLD_NORMAL = "Finance Act - Old Regime Slabs (Age < 60)"
SLAB_OLD_SENIOR = "Finance Act - Old Regime Slabs (Senior Citizen 60-79)"
SLAB_OLD_SUPERSENIOR = "Finance Act - Old Regime Slabs (Super Senior Citizen 80+)"
SLAB_NEW_115BAC = "Sec 115BAC(1A) - New Regime Slabs"

REBATE_87A_OLD = "Sec 87A - Tax Rebate (Old Regime up to ₹5,00,000)"
REBATE_87A_NEW = "Sec 87A - Tax Rebate (New Regime up to ₹7,00,000)"
REBATE_87A_MARGINAL = "Sec 87A Proviso - New Regime Marginal Relief (₹7,00,000 to ₹7,27,777)"

SURCHARGE_APPLIED = "Finance Act - Surcharge on High Income"
SURCHARGE_MARGINAL_RELIEF = "Finance Act - Marginal Relief on Surcharge"
CESS_4_PERCENT = "Finance Act - Health and Education Cess @ 4%"

SEC_288A_ROUNDING = "Sec 288A - Rounding off Taxable Income to nearest ₹10"
SEC_288B_ROUNDING = "Sec 288B - Rounding off Tax to nearest ₹10"


def format_clause_trail(clauses: Iterable[str]) -> list[str]:
    """Ensures deterministic, deduplicated order of clauses cited."""
    seen = set()
    ordered = []
    for c in clauses:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

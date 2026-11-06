"""
(a) Hand-Coded Deterministic Rules Engine (Project A, Task 2a).

Statutory implementation for FY 2024-25 (AY 2025-26) salaried individuals.
Implements rupee-exact statutory calculations according to the Income-tax Act, 1961
and the Finance (No. 2) Act, 2024.
Serves as ground-truth for testing and comparing against (b) and (c).
"""

from decimal import Decimal, ROUND_HALF_UP
import math
from project_a.common.schema import TaxpayerInput, TaxResult
from project_a.common import clause_refs as cr


def _round_10(val: float | Decimal) -> int:
    """Rounds to nearest multiple of 10 per Section 288A / 288B (half up)."""
    d = Decimal(str(val))
    divided = d / Decimal("10")
    rounded = divided.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(rounded * Decimal("10"))


def compute_hra_exemption(
    basic_salary: float,
    hra_received: float,
    rent_paid: float,
    city_is_metro: bool,
) -> tuple[float, list[str]]:
    """
    Computes House Rent Allowance (HRA) exemption under Section 10(13A) read with Rule 2A.
    Exemption is least of:
      1. Actual HRA received
      2. Rent paid - 10% of basic salary
      3. 50% of basic salary (metro) or 40% (non-metro)
    """
    if hra_received <= 0 or rent_paid <= 0 or basic_salary <= 0:
        return 0.0, []

    ten_pct_basic = 0.10 * basic_salary
    excess_rent = max(0.0, rent_paid - ten_pct_basic)
    if excess_rent <= 0:
        return 0.0, [cr.SEC_10_13A_HRA]

    metro_pct = 0.50 if city_is_metro else 0.40
    cap_by_salary = metro_pct * basic_salary

    exemption = min(hra_received, excess_rent, cap_by_salary)
    return round(exemption, 2), [cr.SEC_10_13A_HRA]


def compute_surcharge_and_marginal_relief(
    taxable_income: float,
    base_tax: float,
    is_new_regime: bool = False,
) -> tuple[float, list[str]]:
    """
    Computes surcharge under Finance Act with statutory marginal relief.
    Thresholds: ₹50L, ₹1Cr, ₹2Cr, ₹5Cr (Old regime caps at 37%, New regime at 25%).
    """
    if taxable_income <= 50_00_000:
        return 0.0, []

    clauses = [cr.SURCHARGE_APPLIED]

    # Determine rate and threshold
    if taxable_income <= 1_00_00_000:
        rate = 0.10
        thresh = 50_00_000
    elif taxable_income <= 2_00_00_000:
        rate = 0.15
        thresh = 1_00_00_000
    elif taxable_income <= 5_00_00_000:
        rate = 0.25
        thresh = 2_00_00_000
    else:
        rate = 0.25 if is_new_regime else 0.37
        thresh = 5_00_00_000

    raw_surcharge = base_tax * rate

    # Marginal relief on surcharge:
    # Tax + Surcharge cannot exceed: (Tax on threshold) + (Taxable Income - Threshold)
    # We calculate tax on the threshold income
    if thresh == 50_00_000:
        tax_on_thresh = (
            compute_new_regime_slab_tax(50_00_000)
            if is_new_regime
            else compute_old_regime_slab_tax(50_00_000, age=30)
        )
    elif thresh == 1_00_00_000:
        tax_on_thresh = (
            compute_new_regime_slab_tax(1_00_00_000)
            if is_new_regime
            else compute_old_regime_slab_tax(1_00_00_000, age=30)
        )
    elif thresh == 2_00_00_000:
        tax_on_thresh = (
            compute_new_regime_slab_tax(2_00_00_000)
            if is_new_regime
            else compute_old_regime_slab_tax(2_00_00_000, age=30)
        )
    else:
        tax_on_thresh = (
            compute_new_regime_slab_tax(5_00_00_000)
            if is_new_regime
            else compute_old_regime_slab_tax(5_00_00_000, age=30)
        )

    # Add surcharge applicable at threshold if any
    if thresh == 1_00_00_000:
        tax_on_thresh += tax_on_thresh * 0.10
    elif thresh == 2_00_00_000:
        tax_on_thresh += tax_on_thresh * 0.15
    elif thresh == 5_00_00_000:
        tax_on_thresh += tax_on_thresh * 0.25

    excess_income = taxable_income - thresh
    max_allowable_tax_plus_surcharge = tax_on_thresh + excess_income

    if (base_tax + raw_surcharge) > max_allowable_tax_plus_surcharge:
        relief = (base_tax + raw_surcharge) - max_allowable_tax_plus_surcharge
        surcharge = max(0.0, raw_surcharge - relief)
        clauses.append(cr.SURCHARGE_MARGINAL_RELIEF)
    else:
        surcharge = raw_surcharge

    return surcharge, clauses


def compute_old_regime_slab_tax(taxable_income: float, age: int | None = None) -> float:
    """Computes base slab tax under Old Regime based on age category."""
    ti = taxable_income
    if age and age >= 80:
        # Super Senior Citizen
        if ti <= 5_00_000:
            return 0.0
        elif ti <= 10_00_000:
            return (ti - 5_00_000) * 0.20
        else:
            return 1_00_000 + (ti - 10_00_000) * 0.30
    elif age and age >= 60:
        # Senior Citizen
        if ti <= 3_00_000:
            return 0.0
        elif ti <= 5_00_000:
            return (ti - 3_00_000) * 0.05
        elif ti <= 10_00_000:
            return 10_000 + (ti - 5_00_000) * 0.20
        else:
            return 1_10_000 + (ti - 10_00_000) * 0.30
    else:
        # Standard individual (< 60 years)
        if ti <= 2_50_000:
            return 0.0
        elif ti <= 5_00_000:
            return (ti - 2_50_000) * 0.05
        elif ti <= 10_00_000:
            return 12_500 + (ti - 5_00_000) * 0.20
        else:
            return 1_12_500 + (ti - 10_00_000) * 0.30


def compute_new_regime_slab_tax(taxable_income: float) -> float:
    """Computes base slab tax under Section 115BAC(1A) for FY 2024-25."""
    ti = taxable_income
    if ti <= 3_00_000:
        return 0.0
    elif ti <= 7_00_000:
        return (ti - 3_00_000) * 0.05
    elif ti <= 10_00_000:
        return 20_000 + (ti - 7_00_000) * 0.10
    elif ti <= 12_00_000:
        return 50_000 + (ti - 10_00_000) * 0.15
    elif ti <= 15_00_000:
        return 80_000 + (ti - 12_00_000) * 0.20
    else:
        return 1_40_000 + (ti - 15_00_000) * 0.30


def compute_old_regime(inp: TaxpayerInput) -> TaxResult:
    """
    Computes exact old regime tax liability for salaried individual for FY 2024-25.
    Includes Standard Deduction (₹50k), HRA, 80C, 80CCD(1B), 80D, 24(b), 87A rebate,
    surcharge with marginal relief, and 4% cess.
    """
    clauses = []
    gross_salary = max(0.0, float(inp.annual_salary))

    # 1. Standard deduction Section 16(ia) - ₹50,000 for Old Regime
    std_ded = min(gross_salary, 50_000.0)
    clauses.append(cr.SEC_16_IA_OLD)

    # 2. HRA exemption
    hra_exempt = 0.0
    if inp.hra_received and inp.rent_paid and inp.basic_salary:
        city_metro = bool(inp.city_is_metro)
        hra_exempt, hra_clauses = compute_hra_exemption(
            inp.basic_salary, inp.hra_received, inp.rent_paid, city_metro
        )
        clauses.extend(hra_clauses)

    # 3. Section 24(b) - Home Loan Interest for Self-Occupied Property (Cap ₹2,00,000)
    sec_24b = 0.0
    if inp.home_loan_interest_self_occupied > 0:
        sec_24b = min(2_00_000.0, float(inp.home_loan_interest_self_occupied))
        clauses.append(cr.SEC_24B)

    # 4. Chapter VI-A Deductions
    # Section 80C (Cap ₹1,50,000)
    sec_80c = 0.0
    if inp.section_80c > 0:
        sec_80c = min(1_50_000.0, float(inp.section_80c))
        clauses.append(cr.SEC_80C)

    # Section 80CCD(1B) - NPS (Cap ₹50,000)
    sec_80ccd1b = 0.0
    if inp.section_80ccd1b > 0:
        sec_80ccd1b = min(50_000.0, float(inp.section_80ccd1b))
        clauses.append(cr.SEC_80CCD_1B)

    # Section 80D - Health Insurance
    sec_80d = 0.0
    if inp.section_80d_self > 0:
        cap_self = 50_000.0 if (inp.age and inp.age >= 60) else 25_000.0
        ded_self = min(cap_self, float(inp.section_80d_self))
        sec_80d += ded_self
        clauses.append(cr.SEC_80D_SELF)

    if inp.section_80d_parents > 0:
        cap_parents = 50_000.0 if inp.parents_are_senior_citizens else 25_000.0
        ded_parents = min(cap_parents, float(inp.section_80d_parents))
        sec_80d += ded_parents
        clauses.append(cr.SEC_80D_PARENTS)

    total_deductions = std_ded + hra_exempt + sec_24b + sec_80c + sec_80ccd1b + sec_80d
    raw_taxable = max(0.0, gross_salary - total_deductions)

    # Round taxable income to nearest 10 per Section 288A
    taxable_income = float(_round_10(raw_taxable))
    clauses.append(cr.SEC_288A_ROUNDING)

    # Base slab tax
    age = inp.age or 35
    if age >= 80:
        clauses.append(cr.SLAB_OLD_SUPERSENIOR)
    elif age >= 60:
        clauses.append(cr.SLAB_OLD_SENIOR)
    else:
        clauses.append(cr.SLAB_OLD_NORMAL)

    base_tax = compute_old_regime_slab_tax(taxable_income, age=age)

    # Section 87A Rebate: up to ₹12,500 if taxable income <= ₹5,00,000
    rebate_87a = 0.0
    if taxable_income <= 5_00_000.0:
        rebate_87a = min(base_tax, 12_500.0)
        if rebate_87a > 0:
            clauses.append(cr.REBATE_87A_OLD)

    tax_after_rebate = max(0.0, base_tax - rebate_87a)

    # Surcharge and marginal relief
    surcharge = 0.0
    if tax_after_rebate > 0 and taxable_income > 50_00_000:
        surcharge, sur_clauses = compute_surcharge_and_marginal_relief(
            taxable_income, tax_after_rebate, is_new_regime=False
        )
        clauses.extend(sur_clauses)

    # 4% Health & Education Cess
    tax_plus_surcharge = tax_after_rebate + surcharge
    cess = 0.04 * tax_plus_surcharge
    if tax_plus_surcharge > 0:
        clauses.append(cr.CESS_4_PERCENT)

    final_tax_payable = float(_round_10(tax_plus_surcharge + cess))
    clauses.append(cr.SEC_288B_ROUNDING)

    return TaxResult(
        regime="old",
        gross_total_income=round(gross_salary, 2),
        total_deductions=round(total_deductions, 2),
        taxable_income=round(taxable_income, 2),
        tax_before_cess=round(base_tax, 2),
        surcharge=round(surcharge, 2),
        cess=round(cess, 2),
        rebate_87a=round(rebate_87a, 2),
        final_tax_payable=round(final_tax_payable, 2),
        clause_trace=cr.format_clause_trail(clauses),
    )


def compute_new_regime(inp: TaxpayerInput) -> TaxResult:
    """
    Computes exact new regime tax liability for salaried individual for FY 2024-25.
    Includes Standard Deduction (₹75k per Finance Act 2024), 87A rebate (up to ₹7L),
    Section 87A marginal relief (up to ₹7,27,777), surcharge (capped at 25%), and 4% cess.
    """
    clauses = []
    gross_salary = max(0.0, float(inp.annual_salary))

    # Standard deduction Section 16(ia) - ₹75,000 for New Regime (Finance Act 2024)
    std_ded = min(gross_salary, 75_000.0)
    clauses.append(cr.SEC_16_IA_NEW)

    total_deductions = std_ded
    raw_taxable = max(0.0, gross_salary - total_deductions)

    # Round taxable income to nearest 10 per Section 288A
    taxable_income = float(_round_10(raw_taxable))
    clauses.append(cr.SEC_288A_ROUNDING)

    # Slabs under 115BAC(1A)
    clauses.append(cr.SLAB_NEW_115BAC)
    base_tax = compute_new_regime_slab_tax(taxable_income)

    # Section 87A Rebate & Marginal Relief
    rebate_87a = 0.0
    if taxable_income <= 7_00_000.0:
        # Full 100% rebate up to ₹25,000
        rebate_87a = min(base_tax, 25_000.0)
        if rebate_87a > 0:
            clauses.append(cr.REBATE_87A_NEW)
    elif 7_00_000.0 < taxable_income <= 7_27_777.0:
        # Section 87A Marginal Relief under Proviso:
        # Tax before cess shall not exceed the income exceeding ₹7,00,000
        excess_income = taxable_income - 7_00_000.0
        if base_tax > excess_income:
            rebate_87a = base_tax - excess_income
            clauses.append(cr.REBATE_87A_MARGINAL)

    tax_after_rebate = max(0.0, base_tax - rebate_87a)

    # Surcharge and marginal relief
    surcharge = 0.0
    if tax_after_rebate > 0 and taxable_income > 50_00_000:
        surcharge, sur_clauses = compute_surcharge_and_marginal_relief(
            taxable_income, tax_after_rebate, is_new_regime=True
        )
        clauses.extend(sur_clauses)

    # 4% Health & Education Cess
    tax_plus_surcharge = tax_after_rebate + surcharge
    cess = 0.04 * tax_plus_surcharge
    if tax_plus_surcharge > 0:
        clauses.append(cr.CESS_4_PERCENT)

    final_tax_payable = float(_round_10(tax_plus_surcharge + cess))
    clauses.append(cr.SEC_288B_ROUNDING)

    return TaxResult(
        regime="new",
        gross_total_income=round(gross_salary, 2),
        total_deductions=round(total_deductions, 2),
        taxable_income=round(taxable_income, 2),
        tax_before_cess=round(base_tax, 2),
        surcharge=round(surcharge, 2),
        cess=round(cess, 2),
        rebate_87a=round(rebate_87a, 2),
        final_tax_payable=round(final_tax_payable, 2),
        clause_trace=cr.format_clause_trail(clauses),
    )


def recommend_regime(inp: TaxpayerInput) -> TaxResult:
    """Computes both regimes and returns the one with the lowest final tax payable."""
    old = compute_old_regime(inp)
    new = compute_new_regime(inp)
    return old if old.final_tax_payable <= new.final_tax_payable else new

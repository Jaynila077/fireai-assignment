"""
(c) Human-Reviewed and Harmonized LLM-Compiled Rules (Project A, Task 2c).

Produced via `project_a.implementations.llm_compiled` from `data/rules_fy2024_25.md`
and reviewed by a human engineer to match the exact schema signatures of `schema.py`.
Runs as pure, deterministic Python at inference time with zero LLM API calls.
"""

from decimal import Decimal, ROUND_HALF_UP
from project_a.common.schema import TaxpayerInput, TaxResult
from project_a.common import clause_refs as cr


def _round_10(val: Decimal | float) -> int:
    d = Decimal(str(val))
    divided = d / Decimal("10")
    rounded = divided.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(rounded * Decimal("10"))


def compute_old_slab_tax(taxable_income: float, age: int | None = None) -> float:
    ti = taxable_income
    if age and age >= 80:
        if ti <= 500000:
            return 0.0
        elif ti <= 1000000:
            return (ti - 500000) * 0.20
        else:
            return 100000 + (ti - 1000000) * 0.30
    elif age and age >= 60:
        if ti <= 300000:
            return 0.0
        elif ti <= 500000:
            return (ti - 300000) * 0.05
        elif ti <= 1000000:
            return 10000 + (ti - 500000) * 0.20
        else:
            return 110000 + (ti - 1000000) * 0.30
    else:
        if ti <= 250000:
            return 0.0
        elif ti <= 500000:
            return (ti - 250000) * 0.05
        elif ti <= 1000000:
            return 12500 + (ti - 500000) * 0.20
        else:
            return 112500 + (ti - 1000000) * 0.30


def compute_new_slab_tax(taxable_income: float) -> float:
    ti = taxable_income
    if ti <= 300000:
        return 0.0
    elif ti <= 700000:
        return (ti - 300000) * 0.05
    elif ti <= 1000000:
        return 20000 + (ti - 700000) * 0.10
    elif ti <= 1200000:
        return 50000 + (ti - 1000000) * 0.15
    elif ti <= 1500000:
        return 80000 + (ti - 1200000) * 0.20
    else:
        return 140000 + (ti - 1500000) * 0.30


def compute_surcharge_and_marginal_relief(
    taxable_income: float,
    base_tax: float,
    is_new_regime: bool = False,
) -> tuple[float, list[str]]:
    if taxable_income <= 5000000:
        return 0.0, []

    clauses = [cr.SURCHARGE_APPLIED]
    if taxable_income <= 10000000:
        rate = 0.10
        thresh = 5000000
    elif taxable_income <= 20000000:
        rate = 0.15
        thresh = 10000000
    elif taxable_income <= 50000000:
        rate = 0.25
        thresh = 20000000
    else:
        rate = 0.25 if is_new_regime else 0.37
        thresh = 50000000

    raw_surcharge = base_tax * rate
    tax_on_thresh = compute_new_slab_tax(thresh) if is_new_regime else compute_old_slab_tax(thresh, age=30)
    if thresh == 10000000:
        tax_on_thresh += tax_on_thresh * 0.10
    elif thresh == 20000000:
        tax_on_thresh += tax_on_thresh * 0.15
    elif thresh == 50000000:
        tax_on_thresh += tax_on_thresh * 0.25

    max_tax_plus_surcharge = tax_on_thresh + (taxable_income - thresh)
    if (base_tax + raw_surcharge) > max_tax_plus_surcharge:
        surcharge = max(0.0, max_tax_plus_surcharge - base_tax)
        clauses.append(cr.SURCHARGE_MARGINAL_RELIEF)
    else:
        surcharge = raw_surcharge

    return surcharge, clauses


def compute_old_regime(inp: TaxpayerInput) -> TaxResult:
    clauses = []
    gross_salary = max(0.0, float(inp.annual_salary))
    std_ded = min(gross_salary, 50000.0)
    clauses.append(cr.SEC_16_IA_OLD)

    # HRA
    hra_exempt = 0.0
    if inp.hra_received and inp.rent_paid and inp.basic_salary:
        ten_pct = 0.10 * inp.basic_salary
        excess_rent = max(0.0, inp.rent_paid - ten_pct)
        metro_cap = (0.50 if inp.city_is_metro else 0.40) * inp.basic_salary
        hra_exempt = min(float(inp.hra_received), excess_rent, metro_cap)
        clauses.append(cr.SEC_10_13A_HRA)

    sec_24b = min(200000.0, float(inp.home_loan_interest_self_occupied)) if inp.home_loan_interest_self_occupied > 0 else 0.0
    if sec_24b > 0:
        clauses.append(cr.SEC_24B)

    sec_80c = min(150000.0, float(inp.section_80c)) if inp.section_80c > 0 else 0.0
    if sec_80c > 0:
        clauses.append(cr.SEC_80C)

    sec_80ccd1b = min(50000.0, float(inp.section_80ccd1b)) if inp.section_80ccd1b > 0 else 0.0
    if sec_80ccd1b > 0:
        clauses.append(cr.SEC_80CCD_1B)

    sec_80d = 0.0
    if inp.section_80d_self > 0:
        cap_self = 50000.0 if (inp.age and inp.age >= 60) else 25000.0
        sec_80d += min(cap_self, float(inp.section_80d_self))
        clauses.append(cr.SEC_80D_SELF)
    if inp.section_80d_parents > 0:
        cap_par = 50000.0 if inp.parents_are_senior_citizens else 25000.0
        sec_80d += min(cap_par, float(inp.section_80d_parents))
        clauses.append(cr.SEC_80D_PARENTS)

    total_deductions = std_ded + hra_exempt + sec_24b + sec_80c + sec_80ccd1b + sec_80d
    taxable_income = float(_round_10(max(0.0, gross_salary - total_deductions)))
    clauses.append(cr.SEC_288A_ROUNDING)

    age = inp.age or 35
    if age >= 80:
        clauses.append(cr.SLAB_OLD_SUPERSENIOR)
    elif age >= 60:
        clauses.append(cr.SLAB_OLD_SENIOR)
    else:
        clauses.append(cr.SLAB_OLD_NORMAL)

    base_tax = compute_old_slab_tax(taxable_income, age=age)

    rebate_87a = 0.0
    if taxable_income <= 500000.0:
        rebate_87a = min(base_tax, 12500.0)
        if rebate_87a > 0:
            clauses.append(cr.REBATE_87A_OLD)

    tax_after_rebate = max(0.0, base_tax - rebate_87a)
    surcharge, sur_clauses = compute_surcharge_and_marginal_relief(taxable_income, tax_after_rebate, is_new_regime=False)
    clauses.extend(sur_clauses)

    cess = 0.04 * (tax_after_rebate + surcharge)
    if (tax_after_rebate + surcharge) > 0:
        clauses.append(cr.CESS_4_PERCENT)

    final_tax_payable = float(_round_10(tax_after_rebate + surcharge + cess))
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
    clauses = []
    gross_salary = max(0.0, float(inp.annual_salary))
    std_ded = min(gross_salary, 75000.0)
    clauses.append(cr.SEC_16_IA_NEW)

    taxable_income = float(_round_10(max(0.0, gross_salary - std_ded)))
    clauses.append(cr.SEC_288A_ROUNDING)
    clauses.append(cr.SLAB_NEW_115BAC)

    base_tax = compute_new_slab_tax(taxable_income)

    rebate_87a = 0.0
    if taxable_income <= 700000.0:
        rebate_87a = min(base_tax, 25000.0)
        if rebate_87a > 0:
            clauses.append(cr.REBATE_87A_NEW)
    elif 700000.0 < taxable_income <= 727777.0:
        excess = taxable_income - 700000.0
        if base_tax > excess:
            rebate_87a = base_tax - excess
            clauses.append(cr.REBATE_87A_MARGINAL)

    tax_after_rebate = max(0.0, base_tax - rebate_87a)
    surcharge, sur_clauses = compute_surcharge_and_marginal_relief(taxable_income, tax_after_rebate, is_new_regime=True)
    clauses.extend(sur_clauses)

    cess = 0.04 * (tax_after_rebate + surcharge)
    if (tax_after_rebate + surcharge) > 0:
        clauses.append(cr.CESS_4_PERCENT)

    final_tax_payable = float(_round_10(tax_after_rebate + surcharge + cess))
    clauses.append(cr.SEC_288B_ROUNDING)

    return TaxResult(
        regime="new",
        gross_total_income=round(gross_salary, 2),
        total_deductions=round(std_ded, 2),
        taxable_income=round(taxable_income, 2),
        tax_before_cess=round(base_tax, 2),
        surcharge=round(surcharge, 2),
        cess=round(cess, 2),
        rebate_87a=round(rebate_87a, 2),
        final_tax_payable=round(final_tax_payable, 2),
        clause_trace=cr.format_clause_trail(clauses),
    )


def recommend_regime(inp: TaxpayerInput) -> TaxResult:
    old = compute_old_regime(inp)
    new = compute_new_regime(inp)
    return old if old.final_tax_payable <= new.final_tax_payable else new

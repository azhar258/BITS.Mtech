/* ================================================================

   SINGLE Prolog source file required by the assignment.

   Attributes:
     Income      - Annual income
     CIBIL       - CIBIL score
     Defaults    - yes/no previous loan defaults
     Employment  - employment history in years

   ================================================================ */


/* ================================================================
   HIGH-INCOME PATH
   Income >= Rs. 5,00,000
   ================================================================ */


/* CIBIL >= 750 + no previous default -> APPROVED. */
high_income_decision(Income, CIBIL, no, _Employment, approved) :-
    Income >= 500000,
    CIBIL >= 750.


/* CIBIL >= 750 + previous default + employment >= 1 year
   -> APPROVED (Lower Credit Limit). */
high_income_decision(
    Income, CIBIL, yes, Employment, approved_lower_credit_limit
) :-
    Income >= 500000,
    CIBIL >= 750,
    Employment >= 1.


/* CIBIL >= 750 + previous default + employment < 1 year
   -> REJECTED. */
high_income_decision(
    Income, CIBIL, yes, Employment, rejected
) :-
    Income >= 500000,
    CIBIL >= 750,
    Employment < 1.


/* CIBIL 700-749 + employment >= 1 year
   -> APPROVED (Lower Credit Limit). */
high_income_decision(
    Income, CIBIL, _Defaults, Employment, approved_lower_credit_limit
) :-
    Income >= 500000,
    CIBIL >= 700,
    CIBIL < 750,
    Employment >= 1.


/* CIBIL 700-749 + employment < 1 year
   -> REJECTED. */
high_income_decision(
    Income, CIBIL, _Defaults, Employment, rejected
) :-
    Income >= 500000,
    CIBIL >= 700,
    CIBIL < 750,
    Employment < 1.


/* CIBIL < 700 -> REJECTED. */
high_income_decision(
    Income, CIBIL, _Defaults, _Employment, rejected
) :-
    Income >= 500000,
    CIBIL < 700.


/* ================================================================
   LOW-INCOME PATH
   Income < Rs. 5,00,000
   ================================================================ */


/* Income Rs. 3,00,000 to below Rs. 5,00,000
   and CIBIL >= 750 -> APPROVED (With Co-Signer). */
low_income_decision(
    Income, CIBIL, _Defaults, _Employment, approved_with_cosigner
) :-
    Income >= 300000,
    Income < 500000,
    CIBIL >= 750.


/* Income Rs. 3,00,000 to below Rs. 5,00,000
   and CIBIL < 750 -> REJECTED. */
low_income_decision(
    Income, CIBIL, _Defaults, _Employment, rejected
) :-
    Income >= 300000,
    Income < 500000,
    CIBIL < 750.


/* Income below Rs. 3,00,000 -> REJECTED. */
low_income_decision(
    Income, _CIBIL, _Defaults, _Employment, rejected
) :-
    Income < 300000.


/* ================================================================
   MAIN DECISION PREDICATE
   ================================================================ */


/* High-income applicants use the High-Income Path. */
loan_decision(Income, CIBIL, Defaults, Employment, Decision) :-
    Income >= 500000,
    high_income_decision(
        Income, CIBIL, Defaults, Employment, Decision
    ).


/* Low-income applicants use the Low-Income Path. */
loan_decision(Income, CIBIL, Defaults, Employment, Decision) :-
    Income < 500000,
    low_income_decision(
        Income, CIBIL, Defaults, Employment, Decision
    ).


/* ================================================================
   INPUT VALIDATION
   ================================================================ */


/* Annual income must be a non-negative number. */
valid_income(Income) :-
    number(Income),
    Income >= 0.


/* CIBIL must be a non-negative number. */
valid_cibil(CIBIL) :-
    number(CIBIL),
    CIBIL >= 0.


/* Employment duration must be a non-negative number. */
valid_employment(Employment) :-
    number(Employment),
    Employment >= 0.


/* Previous defaults must be yes or no. */
valid_default(no).
valid_default(yes).


/* ================================================================
   USER INPUT
   ================================================================ */


/* Read and validate Annual Income. */
get_income(Income) :-
    repeat,
    write('Enter Annual Income (numeric value): '),
    read(Input),
    (
        valid_income(Input)
        ->
        Income = Input,
        !
        ;
        write('Invalid income. Please enter a non-negative numeric value.'),
        nl,
        fail
    ).


/* Read and validate CIBIL Score. */
get_cibil(CIBIL) :-
    repeat,
    write('Enter CIBIL Score (numeric value): '),
    read(Input),
    (
        valid_cibil(Input)
        ->
        CIBIL = Input,
        !
        ;
        write('Invalid CIBIL score. Please enter a non-negative numeric value.'),
        nl,
        fail
    ).


/* Read and validate previous-default status. */
get_defaults(Defaults) :-
    repeat,
    write('Enter Any Past Defaults? (yes/no): '),
    read(Input),
    (
        valid_default(Input)
        ->
        Defaults = Input,
        !
        ;
        write('Invalid input. Please enter yes or no.'),
        nl,
        fail
    ).


/* Read and validate employment history. */
get_employment(Employment) :-
    repeat,
    write('Enter Employment History Duration in years (numeric value): '),
    read(Input),
    (
        valid_employment(Input)
        ->
        Employment = Input,
        !
        ;
        write('Invalid employment duration. Please enter a non-negative numeric value.'),
        nl,
        fail
    ).


/* ================================================================
   OUTPUT
   ================================================================ */


/* Display the exact human-readable final decision. */
display_decision(approved) :-
    write('Final Decision: APPROVED'),
    nl.

display_decision(approved_lower_credit_limit) :-
    write('Final Decision: APPROVED (Lower Credit Limit)'),
    nl.

display_decision(approved_with_cosigner) :-
    write('Final Decision: APPROVED (With Co-Signer)'),
    nl.

display_decision(rejected) :-
    write('Final Decision: REJECTED'),
    nl.


/* ================================================================
   MAIN USER INTERFACE
   ================================================================ */


/* Main predicate requested in the assignment. */
evaluate_loan :-
    nl,
    write('--- Bank Loan Approval Expert System ---'),
    nl,

    get_income(Income),
    get_cibil(CIBIL),
    get_defaults(Defaults),
    get_employment(Employment),

    loan_decision(
        Income, CIBIL, Defaults, Employment, Decision
    ),

    display_decision(Decision).

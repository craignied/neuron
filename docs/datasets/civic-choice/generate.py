#!/usr/bin/env python3
"""Generate the fictional Civic Choice binary-classification dataset.

This dataset exists only to demonstrate neuron's modeling and evaluation
workflow. It does not describe real voters or real political parties.
"""

import argparse
import csv
import math
import random
from pathlib import Path


ETHNICITIES = (
    "caucasian",
    "african-american",
    "hispanic",
    "asian",
    "other",
)
EDUCATION = (
    "not-high-school",
    "high-school",
    "college",
    "graduate-professional",
)


def logistic(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def choose_education(rng):
    draw = rng.random()
    if draw < 0.10:
        return EDUCATION[0]
    if draw < 0.45:
        return EDUCATION[1]
    if draw < 0.78:
        return EDUCATION[2]
    return EDUCATION[3]


def make_row(rng):
    age = rng.randint(18, 84)
    education = choose_education(rng)
    education_index = EDUCATION.index(education)

    # Income is realistic enough to make the raw data recognizable, but is
    # clipped so a few extreme draws do not dominate scaling.
    log_income = rng.normalvariate(10.55 + 0.20 * education_index, 0.52)
    income = int(round(min(220_000, max(15_000, math.exp(log_income))) / 500) * 500)

    married_probability = min(0.78, max(0.12, 0.18 + 0.011 * (age - 18)))
    married = rng.random() < married_probability
    employed_probability = 0.88
    if age < 23:
        employed_probability = 0.68
    elif age >= 67:
        employed_probability = 0.35
    employed = rng.random() < employed_probability

    home_probability = 0.12 + 0.008 * (age - 18) + 0.0000022 * (income - 40_000)
    owns_home = rng.random() < min(0.88, max(0.08, home_probability))

    # These three variables are deliberately null in the outcome mechanism.
    # Ethnicity is also generated independently of the socioeconomic variables:
    # the example must not encode a hidden protected-attribute proxy story.
    ethnicity = rng.choice(ETHNICITIES)
    ice_cream = rng.choice(("vanilla", "chocolate"))
    owns_car = rng.random() < 0.62

    score = -0.10

    # A strong U-shaped age response: the two tails favor Party A while the
    # middle favors Party B. No single coefficient on raw age can express two
    # changes of direction; a small hidden layer can build the two boundaries.
    if age < 30:
        score += 1.45
    elif age < 55:
        score -= 1.25
    elif age < 68:
        score += 0.10
    else:
        score += 1.20

    # A second non-monotone response, deliberately on broad, well-populated
    # bands. Middle incomes favor Party A and both tails favor Party B.
    if income < 42_000:
        score -= 1.15
    elif income < 105_000:
        score += 1.00
    else:
        score -= 1.05

    score += {
        "not-high-school": -0.60,
        "high-school": -0.20,
        "college": 0.35,
        "graduate-professional": 0.65,
    }[education]

    # A soft XOR: matching marital/home states favor Party A and mismatched
    # states favor Party B. Each variable still has a modest marginal signal
    # because their prevalences are not exactly balanced.
    score += 0.85 if married == owns_home else -0.85

    # Conventional main effect retained for the explanatory regression.
    score += 0.35 if employed else -0.35

    # Irreducible heterogeneity prevents perfect classification.
    score += rng.normalvariate(0.0, 0.30)
    party_a = rng.random() < logistic(score)

    return (
        age,
        income,
        ethnicity,
        education,
        "married" if married else "single",
        "employed" if employed else "unemployed",
        ice_cream,
        "yes" if owns_car else "no",
        "yes" if owns_home else "no",
        "party-a" if party_a else "party-b",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the fictional, nonlinear Civic Choice dataset."
    )
    parser.add_argument("-o", "--output", default="civic_choice.csv")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    if args.rows < 100:
        parser.error("--rows must be at least 100")

    rng = random.Random(args.seed)
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "age",
                "annual_income",
                "ethnicity",
                "highest_education",
                "marital_status",
                "employment_status",
                "preferred_ice_cream",
                "owns_car",
                "owns_home",
                "choice",
            )
        )
        for _ in range(args.rows):
            writer.writerow(make_row(rng))

    print(
        f"Wrote {args.rows} fictional observations to {output} "
        f"(seed {args.seed})."
    )
    print(
        "Outcome mechanism: nonlinear age/income/education plus interactions; "
        "ethnicity, ice cream, and car ownership are null."
    )


if __name__ == "__main__":
    main()

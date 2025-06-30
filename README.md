# Trade Scoring Engine

## Overview

This repository implements a scoring and classification framework for option and futures trades based on structured decision matrices. The purpose is to evaluate each trade's alignment with specific strategic portfolios (Portfolio1 or Portfolio2) based on predefined trade characteristics.

This tool was developed in a Brattle project. The version here removes confidential client information and inputs retains the underlying logic and methodology.

---

## Objectives

* Classify trades by applying deterministic rules derived from time-of-day matrices.
* Score each trade based on a combination of trade metadata (e.g., option type, tenor, expiration, transaction direction).
* Distinguish between strategic portfolio alignments based on these rules.
* Integrate profit calculations for performance attribution.

---

## Methodology

1. **Matrix Construction**:
   Time-of-day scoring matrices (Morning, Afternoon, Overnight) are parsed from structured Excel files. Each matrix contains classification rules based on trade type, tenor, expiration bucket, and transaction direction.

2. **Trade Preprocessing**:
   Trade records are processed to extract relevant attributes:

   * **Weekly vs. Monthly Option**: Identified via regex on trade description.
   * **Expiration Mapping**: Trade dates are mapped to option expiration dates.
   * **Bucket Assignment**: Trades are assigned to `M1`, `M2`, `Weekly`, or `Futures` buckets depending on maturity relative to expiration.
   * **Option Type**: Trades are tagged as `Call`, `Put`, or `Futures`.
   * **Tenor and Direction**: Derived from internal trade metadata.

3. **Scoring Engine**:
   Each trade is scored by matching it to the appropriate matrix based on its time interval (`Morning`, `Afternoon`, etc.). If both "Rule" fields match a target portfolio, a `High` certainty is assigned. Otherwise, secondary "Principle" rules are used for a `Medium` certainty classification.

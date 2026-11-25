"""Past-only feature engineering (Role 3 — Harsh Kumar).

Public API
----------
Aggregates (card-level, merchant-level, combo):
    past_rolling_aggregates
    add_velocity_features
    add_transaction_count_in_window
    add_merchant_aggregates
    add_card_merchant_aggregates
    add_amount_deviation_features
    add_high_spend_spike_flag
    build_card_features

Time features:
    add_time_features
    add_all_time_features

Amount features (stateless):
    log1p_amount
    amount_bins
    amount_bin_codes
    round_amount_flag
    cents_component
    build_amount_features

Amount features (stateful — fit on train only):
    AmountFeaturizer
"""

from .aggregates import (
    past_rolling_aggregates,
    add_velocity_features,
    add_transaction_count_in_window,
    add_merchant_aggregates,
    add_card_merchant_aggregates,
    add_amount_deviation_features,
    add_high_spend_spike_flag,
    build_card_features,
)
from .time_features import add_time_features, add_all_time_features
from .amount_features import (
    log1p_amount,
    amount_bins,
    amount_bin_codes,
    round_amount_flag,
    cents_component,
    build_amount_features,
    AmountFeaturizer,
)

__all__ = [
    # aggregates
    "past_rolling_aggregates",
    "add_velocity_features",
    "add_transaction_count_in_window",
    "add_merchant_aggregates",
    "add_card_merchant_aggregates",
    "add_amount_deviation_features",
    "add_high_spend_spike_flag",
    "build_card_features",
    # time
    "add_time_features",
    "add_all_time_features",
    # amount (stateless)
    "log1p_amount",
    "amount_bins",
    "amount_bin_codes",
    "round_amount_flag",
    "cents_component",
    "build_amount_features",
    # amount (stateful)
    "AmountFeaturizer",
]

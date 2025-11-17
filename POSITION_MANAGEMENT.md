# Position Management System - Documentation

## Overview
This trading system now implements **cumulative position sizing** based on multiple trading signals, following professional trading practices including pyramiding and risk management.

## Key Features

### 1. Cumulative Position Sizing (Pyramiding)
- **Multiple Buy Signals Add Up**: If both IBS Reversion AND Gap and Go strategies trigger buy signals on the same day, the position becomes 2×BP (two base positions).
- **Position Tracking**: Positions are tracked as a single integer:
  - **Positive values** = Long positions (e.g., +1, +2, +3)
  - **Negative values** = Short positions (e.g., -1, -2, -3)
  - **Zero** = Flat (no position)

### 2. Opposite Signal Handling (Position Reduction)
Following professional algorithmic trading practices, opposite signals **reduce** existing positions rather than reversing or being ignored.

**Examples:**
- **Scenario 1**: You're at +2 long position, and a sell signal (SP=1) occurs
  - Result: Position reduces to +1 long
  - You've partially taken profit while maintaining exposure
  
- **Scenario 2**: You're at +1 long position, and a sell signal (SP=1) occurs
  - Result: Position becomes 0 (flat)
  - Position fully closed
  
- **Scenario 3**: You're at 0 (flat), and a sell signal (SP=1) occurs
  - Result: Position becomes -1 short
  - New short position opened
  
- **Scenario 4**: You're at -2 short position, and a buy signal (BP=1) occurs
  - Result: Position reduces to -1 short
  - Partial covering of short position

### 3. Position Limits (Risk Management)
- **Default Maximum**: ±3 positions
- **Prevents Over-Exposure**: Even if 5 buy signals trigger simultaneously, position is capped at +3
- **Configurable**: Can be adjusted via the `max_position` parameter in `strategy_from_logic()`

```python
# Example: Set custom position limit
strategy_from_logic(df, rules, params, max_position=5)
```

### 4. Exit Logic
When exit signals (Stop Loss, Take Profit, or Exit) trigger:
- **All positions are closed simultaneously** at the exit price
- Each position entry generates a separate trade record with its own PnL
- Priority: Stop Loss > Take Profit > Regular Exit

## Professional Trading Rationale

### Why Cumulative Positions?
✅ **Multiple Confirmations = Higher Confidence**: When multiple strategies agree, market conviction is stronger  
✅ **Pyramiding Strategy**: A recognized professional technique for trend-following  
✅ **Maximizes Returns**: Capitalizes on high-probability setups  

### Why Reduce (Not Reverse or Ignore)?
✅ **Flexibility**: Allows partial profit-taking while maintaining exposure  
✅ **Risk Management**: Gradually adjusts exposure based on changing signals  
✅ **Professional Standard**: Most algorithmic trading systems use this approach  
✅ **Avoids Whipsaws**: Prevents rapid position reversals during choppy markets  

**Alternative Approaches (Not Implemented):**
- ❌ **Ignore Opposite Signals**: Misses opportunities to take profits
- ❌ **Full Reversal**: Too aggressive, increases transaction costs and whipsaw risk
- ❌ **Exit Only**: Less flexible, binary decision-making

### Why Position Limits?
✅ **Prevents Over-Leverage**: Limits maximum exposure even when all strategies align  
✅ **Standardized Risk**: Each strategy adds 1 position unit, making risk calculations consistent  
✅ **Portfolio Protection**: Ensures no single trade dominates the portfolio  

## Example Walkthrough

### Multi-Day Position Evolution

| Day | Event | Position Before | Position After | Notes |
|-----|-------|-----------------|----------------|-------|
| 1 | IBS Reversion Buy (BP=1) | 0 | +1 | Opened long |
| 2 | Gap and Go Buy (BP=1) | +1 | +2 | Added to position (pyramiding) |
| 3 | No signals | +2 | +2 | Holding |
| 4 | Noise Area Breakout Sell (SP=1) | +2 | +1 | Reduced position (opposite signal) |
| 5 | Take Profit triggered | +1 | 0 | Closed all positions |

**PnL Calculation**: Two separate trades recorded (one for each entry)

## Code Changes Summary

### Modified Files
1. **`strategy.py`**:
   - Changed `position` from dict to `current_position` (integer)
   - Added `position_entries` list to track individual entries
   - Implemented position delta logic for cumulative sizing
   - Added position limit enforcement (`max_position`)
   - Refactored exit logic to close all positions
   - Updated to generate separate trade records per position unit

### Configuration
Add to your Excel config or Python dict:
```python
config = {
    "param_map": {
        "BP": 1,  # Base Position for Buy signals
        "SP": 1,  # Base Position for Sell signals
        # ... other parameters
    },
    # Optionally configure max_position in strategy call
}
```

### Usage in Main Script
```python
from strategy import strategy_from_logic, parse_strategy_logic

rules = parse_strategy_logic(logic_df)
params = {"BP": 1, "SP": 1}

# Default max_position = 3
results = strategy_from_logic(df, rules, params)

# Custom max_position
results = strategy_from_logic(df, rules, params, max_position=5)
```

## Testing Recommendations

1. **Test Cumulative Entries**: Create scenarios with multiple simultaneous buy signals
2. **Test Opposite Signals**: Verify position reduces correctly (not reverses)
3. **Test Position Limits**: Ensure positions cap at max_position
4. **Test Exit Logic**: Confirm all positions close together with correct PnL
5. **Backtest Comparison**: Compare new vs. old logic on historical data

## Risk Considerations

⚠️ **Higher Exposure**: Cumulative positions mean larger capital allocation when multiple strategies align  
⚠️ **Correlation Risk**: Multiple strategies might trigger due to correlated signals, not independent confirmations  
⚠️ **Transaction Costs**: More position adjustments = higher commissions/slippage  

**Mitigation Strategies:**
- Keep `max_position` conservative (3-5)
- Monitor correlation between strategies
- Adjust BP/SP sizes based on account size
- Implement per-trade risk limits (e.g., risk 1% per BP)

## References
- [Position Sizing in Trend Following](https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/)
- [How to Adjust Trade Positions](https://www.dummies.com/article/business-careers-money/personal-finance/investing/general-investing/how-to-adjust-trade-positions-189963/)
- [Concentrate on Position Sizing](https://systematicindividualinvestor.com/2018/07/03/concentrate-on-position-sizing/)

---
**Version**: 1.0  
**Last Updated**: October 27, 2025  
**Author**: Trading Strategy Optimizer Team



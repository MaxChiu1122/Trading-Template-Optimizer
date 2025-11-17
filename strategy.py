import pandas as pd
from typing import List, Dict
from collections import defaultdict


def parse_strategy_logic(df_logic: pd.DataFrame) -> Dict[str, Dict]:
    """
    Parse the strategy logic table into a rule map for evaluation.
    Returns a dict mapping rule keys to dicts containing conditions and position info.
    
    Each row in the logic table creates a separate rule entry, allowing duplicate conditions
    to trigger multiple times and add multiple positions.
    """
    rule_map = defaultdict(list)
    
    # Find the position column (it might be named "Position", "Position (Value/Param)", etc.)
    position_col = None
    for col in df_logic.columns:
        if "Position" in str(col):
            position_col = col
            break

    # Process rows grouped by Rule Type and Action at
    # Create a separate rule entry for each row that starts a new rule
    grouped = df_logic.groupby(["Rule Type", "Action at"], sort=False)
    
    for (rule_type, action_at), group_df in grouped:
        key = f"{rule_type.strip()}_{action_at.strip()}"
        
        # Process each row in the group
        # Create a separate rule entry for EACH row that is not a continuation
        group_df_list = list(group_df.iterrows())
        for row_idx, (idx, row) in enumerate(group_df_list):
            # Check if this row is a continuation of a previous row's rule
            # A row is a continuation if the IMMEDIATELY previous row in the same group has AND/OR logic
            if row_idx > 0:
                # Get the immediately previous row in this group
                prev_idx, prev_row = group_df_list[row_idx - 1]
                prev_logic = str(prev_row["Logic Type"]).strip().upper() if pd.notna(prev_row["Logic Type"]) else ""
                if prev_logic in {"AND", "OR"}:
                    # This row continues the previous row's rule, skip creating a new rule entry
                    continue
            
            expr_parts = []
            position_info = None
            
            # Build conditions starting from this row
            # Include subsequent rows that have AND/OR logic (until END or empty)
            rows_to_process = group_df_list[row_idx:]
            
            for sub_idx, sub_row in rows_to_process:
                col_a = str(sub_row["Column A"]).strip()
                op = str(sub_row["Operator"]).strip()
                col_b = str(sub_row["Column B / Value"]).strip()
                logic = str(sub_row["Logic Type"]).strip().upper() if pd.notna(sub_row["Logic Type"]) else ""
                
                # Extract position information (take the last non-null value in the chain)
                if position_col and pd.notna(sub_row[position_col]) and str(sub_row[position_col]).strip():
                    position_info = str(sub_row[position_col]).strip()

                # Determine if col_b is a value or a column
                # Special case: if col_a == col_b and op is "==", always True
                if col_a == col_b and op == "==":
                    cond = "(True)"  # Always true condition (e.g., Close == Close)
                elif col_b.replace(".", "", 1).isdigit():
                    cond = f"(row['{col_a}'] {op} {col_b})"
                else:
                    cond = f"(row['{col_a}'] {op} row['{col_b}'])"

                expr_parts.append(cond)

                # Add logical operator if not END
                if logic in {"AND", "OR"}:
                    expr_parts.append(logic.lower())
                elif logic not in {"", "END"}:
                    print(f"⚠️ Unknown logic type: '{logic}' in row {sub_idx}")

                # If we hit END or empty logic, this completes the rule
                if logic in {"", "END"}:
                    break
            
            # Remove trailing logic operator if present
            if expr_parts and expr_parts[-1] in {"and", "or"}:
                expr_parts = expr_parts[:-1]

            # Create a separate rule entry for this row, allowing duplicates
            # Add a unique rule_id to track each rule separately, even if conditions are identical
            conditions_str = " ".join(expr_parts)
            rule_entry = {
                "conditions": conditions_str,
                "position": position_info,
                "rule_id": f"{key}_{row_idx}_{idx}"  # Unique identifier for this rule
            }
            rule_map[key].append(rule_entry)

    return rule_map


def evaluate_conditions(row: pd.Series, conditions: str) -> bool:
    """
    Evaluate if a set of conditions is satisfied for a given row.
    """
    try:
        expr = conditions.strip()
        
        # Fast path: if condition is explicitly "(True)", return True immediately
        if expr == "(True)":
            return True
        # Fast path: if condition is explicitly "(False)", return False immediately
        if expr == "(False)":
            return False
        
        # Ensure all row[...] are scalars, not Series
        # If any value is a Series, use .item() if length 1, else warn and skip
        class SafeRow(dict):
            def __getitem__(self, key):
                val = super().__getitem__(key)
                if isinstance(val, pd.Series):
                    if len(val) == 1:
                        return val.item()
                    else:
                        print(f"[Strategy] Ambiguous value for '{key}' in row: Series of length {len(val)}. Skipping.")
                        raise ValueError(f"Ambiguous value for '{key}' in row.")
                return val
        safe_row = SafeRow(row)
        result = eval(expr, {"row": safe_row, "__builtins__": {}}, {"True": True, "False": False})
        return bool(result)
    except Exception as e:
        print(f"❌ Evaluation error: {e}, expr: {expr}")
        return False


def strategy_from_logic(df: pd.DataFrame, rules: Dict[str, List[Dict]], params: Dict = None, max_position: int = 3, force_same_day_exit: bool = False) -> pd.DataFrame:
    """
    Apply strategy logic to a DataFrame and return a DataFrame of trades with PnL and triggers.
    
    New Trading Logic:
    - For each row, collect all buy/sell entry signals
    - Buy signals count as positive positions (based on size)
    - Sell signals count as negative positions (based on size)
    - Aggregate signals by action (Buy/Sell) and price
    - The net position size at that row (with that action price) is the trade made at that row
    
    Args:
        force_same_day_exit: Not used in new logic (kept for compatibility)
    """
    results = []
    
    # Default parameters if not provided
    if params is None:
        params = {}
    
    def get_position_size(position_info: str) -> int:
        """Get position size from position info, supporting both fixed values and parameters."""
        if not position_info:
            return 1  # Default to 1 position
        
        # Check if it's a parameter (like "BP", "SP")
        if position_info in params:
            return int(params[position_info])
        
        # Check if it's a fixed number
        try:
            return int(float(position_info))
        except (ValueError, TypeError):
            return 1  # Default to 1 if can't parse
    
    def check_exit_conditions(row, open_positions, rules):
        """Check exit conditions and return (exit_triggered, exit_price, stop, take_profit)."""
        if not open_positions:
            return False, None, False, False
        
        exit_triggered = False
        exit_price = None
        stop = False
        take_profit = False
        
        # Determine position direction from open positions
        has_long = any(p["action"] == "Buy" for p in open_positions)
        has_short = any(p["action"] == "Sell" for p in open_positions)
        
        # Check Stop Loss (highest priority)
        if has_long:
            stop_prefix = "StopLoss-long"
            for rule_key in rules:
                if rule_key.startswith(stop_prefix):
                    for rule_info in rules[rule_key]:
                        try:
                            if evaluate_conditions(row, rule_info["conditions"]):
                                _, action_at = rule_key.split("_", 1)
                                exit_price = row.get(action_at, row["Close"])
                                stop = True
                                exit_triggered = True
                                break
                        except Exception:
                            pass
                    if stop:
                        break
        
        if not exit_triggered and has_short:
            stop_prefix = "StopLoss-short"
            for rule_key in rules:
                if rule_key.startswith(stop_prefix):
                    for rule_info in rules[rule_key]:
                        try:
                            if evaluate_conditions(row, rule_info["conditions"]):
                                _, action_at = rule_key.split("_", 1)
                                exit_price = row.get(action_at, row["Close"])
                                stop = True
                                exit_triggered = True
                                break
                        except Exception:
                            pass
                    if stop:
                        break
        
        # Check Take Profit
        if not exit_triggered and has_long:
            tp_prefix = "TakeProfit-long"
            for rule_key in rules:
                if rule_key.startswith(tp_prefix):
                    for rule_info in rules[rule_key]:
                        try:
                            if evaluate_conditions(row, rule_info["conditions"]):
                                _, action_at = rule_key.split("_", 1)
                                exit_price = row.get(action_at, row["Close"])
                                take_profit = True
                                exit_triggered = True
                                break
                        except Exception:
                            pass
                    if take_profit:
                        break
        
        if not exit_triggered and has_short:
            tp_prefix = "TakeProfit-short"
            for rule_key in rules:
                if rule_key.startswith(tp_prefix):
                    for rule_info in rules[rule_key]:
                        try:
                            if evaluate_conditions(row, rule_info["conditions"]):
                                _, action_at = rule_key.split("_", 1)
                                exit_price = row.get(action_at, row["Close"])
                                take_profit = True
                                exit_triggered = True
                                break
                        except Exception:
                            pass
                    if take_profit:
                        break
        
        # Check Regular Exit
        if not exit_triggered and has_long:
            exit_prefix = "Exit-long"
            exit_rules_found = False
            has_true_condition = False
            first_exit_action_at = None
            
            # First pass: check if any exit rule has "(True)" condition or "Close == Close"
            for rule_key in rules:
                if rule_key.startswith(exit_prefix):
                    exit_rules_found = True
                    if first_exit_action_at is None:
                        _, first_exit_action_at = rule_key.split("_", 1)
                    
                    for rule_info in rules[rule_key]:
                        condition_str = rule_info.get("conditions", "").strip()
                        # Check if condition contains "(True)" or "Close == Close" - always true
                        if "(True)" in condition_str or "Close == Close" in condition_str:
                            has_true_condition = True
                            _, action_at = rule_key.split("_", 1)
                            if first_exit_action_at is None:
                                first_exit_action_at = action_at
                            break
            
            # If we found a "(True)" or "Close == Close" condition, always trigger exit
            if has_true_condition:
                exit_price = row.get(first_exit_action_at, row["Close"])
                exit_triggered = True
            else:
                # Second pass: evaluate conditions normally
                for rule_key in rules:
                    if rule_key.startswith(exit_prefix):
                        for rule_info in rules[rule_key]:
                            try:
                                if evaluate_conditions(row, rule_info["conditions"]):
                                    _, action_at = rule_key.split("_", 1)
                                    exit_price = row.get(action_at, row["Close"])
                                    exit_triggered = True
                                    break
                            except Exception:
                                pass
                        if exit_triggered:
                            break
                
                # If exit rules exist but didn't trigger, use first action_at
                if not exit_triggered and exit_rules_found and first_exit_action_at:
                    exit_price = row.get(first_exit_action_at, row["Close"])
                    exit_triggered = True
        
        if not exit_triggered and has_short:
            exit_prefix = "Exit-short"
            exit_rules_found = False
            has_true_condition = False
            first_exit_action_at = None
            
            # First pass: check if any exit rule has "(True)" condition or "Close == Close"
            for rule_key in rules:
                if rule_key.startswith(exit_prefix):
                    exit_rules_found = True
                    if first_exit_action_at is None:
                        _, first_exit_action_at = rule_key.split("_", 1)
                    
                    for rule_info in rules[rule_key]:
                        condition_str = rule_info.get("conditions", "").strip()
                        # Check if condition contains "(True)" or "Close == Close" - always true
                        if "(True)" in condition_str or "Close == Close" in condition_str:
                            has_true_condition = True
                            _, action_at = rule_key.split("_", 1)
                            if first_exit_action_at is None:
                                first_exit_action_at = action_at
                            break
            
            # If we found a "(True)" or "Close == Close" condition, always trigger exit
            if has_true_condition:
                exit_price = row.get(first_exit_action_at, row["Close"])
                exit_triggered = True
            else:
                # Second pass: evaluate conditions normally
                for rule_key in rules:
                    if rule_key.startswith(exit_prefix):
                        for rule_info in rules[rule_key]:
                            try:
                                if evaluate_conditions(row, rule_info["conditions"]):
                                    _, action_at = rule_key.split("_", 1)
                                    exit_price = row.get(action_at, row["Close"])
                                    exit_triggered = True
                                    break
                            except Exception:
                                pass
                        if exit_triggered:
                            break
                
                # If exit rules exist but didn't trigger, use first action_at
                if not exit_triggered and exit_rules_found and first_exit_action_at:
                    exit_price = row.get(first_exit_action_at, row["Close"])
                    exit_triggered = True
        
        return exit_triggered, exit_price, stop, take_profit
    
    def has_always_true_exit(rules):
        """Check if any exit rule has an always-true condition (Close == Close or (True))."""
        for rule_key in rules:
            if rule_key.startswith("Exit-long") or rule_key.startswith("Exit-short"):
                for rule_info in rules[rule_key]:
                    condition_str = rule_info.get("conditions", "").strip()
                    # Check if condition contains "(True)" or "Close == Close" - always true
                    if "(True)" in condition_str or "Close == Close" in condition_str:
                        return True
        return False
    
    # Track open positions: List of {action, entry_price, entry_date, position_size}
    open_positions = []
    
    # Check if exit conditions are always true (for same-day exit logic)
    exit_is_always_true = has_always_true_exit(rules)
    
    # Process each row in the dataframe
    for i in range(len(df)):
        row = df.iloc[i]
        date = row["Date"]
        
        # Step 1: Check for exit signals first (before new entries)
        exit_triggered, exit_price, stop, take_profit = check_exit_conditions(row, open_positions, rules)
        
        # Step 2: Close positions if exit triggered
        if exit_triggered and exit_price is not None:
            # Close all open positions
            while open_positions:
                pos = open_positions.pop(0)
                entry_action = pos["action"]
                entry_price = pos["entry_price"]
                entry_date = pos["entry_date"]
                pos_size = pos["position_size"]
                
                # Calculate PnL
                if entry_action == "Buy":
                    pnl = (exit_price - entry_price) * pos_size
                else:  # Sell
                    pnl = (entry_price - exit_price) * pos_size
                
                results.append({
                    "EntryDate": entry_date,
                    "ExitDate": date,
                    "Action": entry_action,
                    "Entry": entry_price,
                    "Exit": exit_price,
                    "PositionSize": pos_size,
                    "Stop Triggered": stop,
                    "Take Profit Triggered": take_profit,
                    "PnL": pnl
                })
        
        # Step 3: Collect and process new entry signals for this row
        entry_signals = []  # List of (action, entry_field, entry_price, position_size) tuples
        
        for rule_key in rules:
            if rule_key.startswith("Enter-"):
                rule_type, entry_field_candidate = rule_key.split("_")
                
                # Check each rule in the group
                for rule_idx, rule_info in enumerate(rules[rule_key]):
                    try:
                        triggered = evaluate_conditions(row, rule_info["conditions"])
                    except Exception:
                        continue
                    
                    if triggered:
                        action = None
                        if "Buy" in rule_type:
                            action = "Buy"
                        elif "Sell" in rule_type:
                            action = "Sell"
                        
                        if action:
                            entry_field = entry_field_candidate
                            entry_price = row.get(entry_field, row["Open"])
                            position_size = get_position_size(rule_info.get("position"))
                            # Only add signals with non-zero position size
                            if position_size > 0:
                                entry_signals.append((action, entry_field, entry_price, position_size))
        
        # Step 4: Aggregate signals by action type and price
        # Buy signals count as positive positions, Sell signals count as negative positions
        # First, aggregate by action type to support same rule type position addition
        action_groups = {"Buy": 0, "Sell": 0}  # Track total position size by action
        
        for action, entry_field, entry_price, position_size in entry_signals:
            if action == "Buy":
                action_groups["Buy"] += position_size
            else:  # Sell
                action_groups["Sell"] += position_size
        
        # Step 5: Merge new positions with existing open positions of the same type
        # Calculate current net position
        current_net_position = 0
        for pos in open_positions:
            if pos["action"] == "Buy":
                current_net_position += pos["position_size"]
            else:  # Sell
                current_net_position -= pos["position_size"]
        
        # Calculate new net position after adding signals
        new_net_position = current_net_position + action_groups["Buy"] - action_groups["Sell"]
        
        # Apply position limit
        if new_net_position > max_position:
            new_net_position = max_position
        elif new_net_position < -max_position:
            new_net_position = -max_position
        
        # Calculate the change in position
        position_change = new_net_position - current_net_position
        
        if position_change != 0:
            # Handle position reduction first (closing/reducing existing positions)
            if (current_net_position > 0 and position_change < 0) or (current_net_position < 0 and position_change > 0):
                # We're reducing existing positions
                remaining_reduction = abs(position_change)
                
                # Process positions in order to close/reduce them
                positions_to_remove = []
                for pos_idx, pos in enumerate(open_positions):
                    if remaining_reduction <= 0:
                        break
                    
                    # Check if this position is opposite to the change direction
                    if (position_change < 0 and pos["action"] == "Buy") or (position_change > 0 and pos["action"] == "Sell"):
                        if pos["position_size"] <= remaining_reduction:
                            # Close this position entirely
                            remaining_reduction -= pos["position_size"]
                            positions_to_remove.append(pos_idx)
                        else:
                            # Reduce this position
                            pos["position_size"] -= remaining_reduction
                            remaining_reduction = 0
                
                # Remove closed positions (in reverse order to maintain indices)
                for pos_idx in reversed(positions_to_remove):
                    open_positions.pop(pos_idx)
            
            # Recalculate net position after reduction
            current_net_after_reduction = 0
            for pos in open_positions:
                if pos["action"] == "Buy":
                    current_net_after_reduction += pos["position_size"]
                else:  # Sell
                    current_net_after_reduction -= pos["position_size"]
            
            # Calculate remaining position change needed
            remaining_position_change = new_net_position - current_net_after_reduction
            
            # Handle position addition (adding new positions or increasing existing ones)
            # Only add positions if we're actually increasing positions of the same type
            if remaining_position_change > 0 and current_net_after_reduction >= 0:
                # Adding long positions (we're long or flat, and increasing)
                if action_groups["Buy"] > 0:
                    action = "Buy"
                    # Use weighted average entry price from buy signals
                    buy_signals = [(ep, ps) for a, ef, ep, ps in entry_signals if a == "Buy"]
                    if buy_signals:
                        total_size = sum(ps for _, ps in buy_signals)
                        if total_size > 0:
                            entry_price = sum(ep * ps for ep, ps in buy_signals) / total_size
                        else:
                            entry_price = row.get("Open", row["Close"])
                    else:
                        entry_price = row.get("Open", row["Close"])
                    
                    # Find existing Buy position to merge with, or create new
                    merged = False
                    merged_pos = None
                    for pos in open_positions:
                        if pos["action"] == "Buy":
                            # Merge: update position size and recalculate average entry price
                            old_size = pos["position_size"]
                            old_price = pos["entry_price"]
                            new_size = old_size + abs(remaining_position_change)
                            
                            # Weighted average entry price
                            if new_size > 0:
                                pos["entry_price"] = (old_price * old_size + entry_price * abs(remaining_position_change)) / new_size
                            pos["position_size"] = new_size
                            merged = True
                            merged_pos = pos
                            break
                    
                    if not merged:
                        # Create new position entry
                        new_pos_size = abs(remaining_position_change)
                        open_positions.append({
                            "action": action,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "position_size": new_pos_size
                        })
            
            elif remaining_position_change < 0 and current_net_after_reduction <= 0:
                # Adding short positions (we're short or flat, and increasing shorts)
                if action_groups["Sell"] > 0:
                    action = "Sell"
                    # Use weighted average entry price from sell signals
                    sell_signals = [(ep, ps) for a, ef, ep, ps in entry_signals if a == "Sell"]
                    if sell_signals:
                        total_size = sum(ps for _, ps in sell_signals)
                        if total_size > 0:
                            entry_price = sum(ep * ps for ep, ps in sell_signals) / total_size
                        else:
                            entry_price = row.get("Open", row["Close"])
                    else:
                        entry_price = row.get("Open", row["Close"])
                    
                    # Find existing Sell position to merge with, or create new
                    merged = False
                    for pos in open_positions:
                        if pos["action"] == "Sell":
                            # Merge: update position size and recalculate average entry price
                            old_size = pos["position_size"]
                            old_price = pos["entry_price"]
                            new_size = old_size + abs(remaining_position_change)
                            
                            # Weighted average entry price
                            if new_size > 0:
                                pos["entry_price"] = (old_price * old_size + entry_price * abs(remaining_position_change)) / new_size
                            pos["position_size"] = new_size
                            merged = True
                            break
                    
                    if not merged:
                        # Create new position entry
                        open_positions.append({
                            "action": action,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "position_size": abs(remaining_position_change)
                        })
        
        # Step 6: Check exits again AFTER adding new positions ONLY if exit condition is always true
        # This allows positions opened on this row to exit immediately if exit condition is always true (e.g., Close == Close)
        # But for multi-day positions, we don't check exits again - they'll be checked on the next row
        if exit_is_always_true:
            exit_triggered, exit_price, stop, take_profit = check_exit_conditions(row, open_positions, rules)
            
            if exit_triggered and exit_price is not None:
                # Close all open positions
                while open_positions:
                    pos = open_positions.pop(0)
                    entry_action = pos["action"]
                    entry_price = pos["entry_price"]
                    entry_date = pos["entry_date"]
                    pos_size = pos["position_size"]
                    
                    # Calculate PnL
                    if entry_action == "Buy":
                        pnl = (exit_price - entry_price) * pos_size
                    else:  # Sell
                        pnl = (entry_price - exit_price) * pos_size
                    
                    results.append({
                        "EntryDate": entry_date,
                        "ExitDate": date,
                        "Action": entry_action,
                        "Entry": entry_price,
                        "Exit": exit_price,
                        "PositionSize": pos_size,
                        "Stop Triggered": stop,
                        "Take Profit Triggered": take_profit,
                        "PnL": pnl
                    })
    
    # Step 7: Close any remaining open positions at the last available price
    if open_positions:
        last_row = df.iloc[-1]
        last_date = last_row["Date"]
        exit_price = last_row["Close"]
        
        while open_positions:
            pos = open_positions.pop(0)
            entry_action = pos["action"]
            entry_price = pos["entry_price"]
            entry_date = pos["entry_date"]
            pos_size = pos["position_size"]
            
            # Calculate PnL
            if entry_action == "Buy":
                pnl = (exit_price - entry_price) * pos_size
            else:  # Sell
                pnl = (entry_price - exit_price) * pos_size
            
            results.append({
                "EntryDate": entry_date,
                "ExitDate": last_date,
                "Action": entry_action,
                "Entry": entry_price,
                "Exit": exit_price,
                "PositionSize": pos_size,
                "Stop Triggered": False,
                "Take Profit Triggered": False,
                "PnL": pnl
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Aggregate trades with same entry/exit dates, action, and price
        results_df = aggregate_trades(results_df)
    
    return results_df


def aggregate_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trades with the same EntryDate, ExitDate, and Action into single rows.
    This combines multiple position units into a single trade record.
    """
    if trades_df.empty:
        return trades_df
    
    # Group by EntryDate, ExitDate, Action
    agg_dict = {
        "Entry": "mean",  # Average entry price
        "Exit": "mean",   # Average exit price (should be the same for same exit date)
        "PositionSize": "sum",  # Sum up position sizes
        "Stop Triggered": "any",  # True if any position hit stop
        "Take Profit Triggered": "any",  # True if any position hit take profit
        "PnL": "sum"  # Total PnL
    }
    
    aggregated = trades_df.groupby(["EntryDate", "ExitDate", "Action"], as_index=False).agg(agg_dict)
    
    # Sort by EntryDate
    aggregated = aggregated.sort_values("EntryDate").reset_index(drop=True)
    
    return aggregated
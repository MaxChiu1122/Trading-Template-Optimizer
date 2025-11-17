import numpy as np
import pandas as pd

def build_indicators(df, param_map, builder_df=None, talib_df=None):
    try:
        import talib
    except ImportError:
        talib = None
    df = df.copy()
    
    # TA-Lib indicators built FIRST
    if talib is not None and talib_df is not None:
        for idx, row in talib_df.iterrows():
            name = row.get("TA-Lib Name")
            func = row.get("TA-Lib Function")
            in_col = row.get("In order Indicators")
            param_str = row.get("In order Param")
            if not (name and func):
                continue
            input_cols = [c.strip() for c in str(in_col).split(",") if c.strip()] if in_col else []
            input_series = []
            if input_cols and all(c in df.columns for c in input_cols):
                input_series = [df[c] for c in input_cols]
            else:
                continue
            param_keys = [p.strip() for p in str(param_str).split(",") if p.strip()] if param_str else []
            param_vals = [param_map.get(k, k) for k in param_keys]
            for i, v in enumerate(param_vals):
                try:
                    param_vals[i] = float(v)
                except Exception:
                    continue
            try:
                talib_func = getattr(talib, func)
            except Exception:
                continue
            try:
                if len(input_series) == 1:
                    out = talib_func(input_series[0], *param_vals)
                else:
                    out = talib_func(*input_series, *param_vals)
                out_names = [n.strip() for n in name.split(",")]
                assigned = False
                if isinstance(out, (tuple, list)) and len(out_names) == len(out):
                    for n, o in zip(out_names, out):
                        if hasattr(o, '__len__') and not isinstance(o, str) and len(o) == len(df):
                            df[n] = o
                            assigned = True
                if not assigned:
                    if hasattr(out, '__len__') and not isinstance(out, str) and len(out) == len(df):
                        df[name] = out
            except Exception:
                continue
    
    # Arithmetic indicators with combination logic built SECOND (after TA-Lib)
    if builder_df is not None and 'Combination' in builder_df.columns:
        for name, group in builder_df.groupby('Indicator Name', sort=False):
            result = None
            prev_comb = None
            for idx, row in group.iterrows():
                # print(f"[DEBUG] Indicator: {name}, Step {idx}, ind_a: {row.get('Indicator A')}, op: {row.get('Operator')}, val/param: {row.get('Value / Param')}, comb: {row.get('Combination')}")
                ind_a = row.get("Indicator A")
                op = row.get("Operator")
                val_or_param = row.get("Value / Param")
                comb = str(row.get("Combination", "END")).strip().upper()
                if not (ind_a and op and val_or_param):
                    continue
                # Get left operand
                left = df[str(ind_a)] if str(ind_a) in df.columns else None
                if left is None:
                    print(f"[WARNING] Indicator '{name}': Column '{ind_a}' not found in dataframe. Available columns: {list(df.columns)}")
                    continue
                # Get right operand (parameter or column)
                val = param_map.get(str(val_or_param), val_or_param)
                if str(val) in df.columns:
                    val_operand = df[str(val)]
                else:
                    try:
                        val_operand = float(val)
                    except Exception:
                        print(f"[WARNING] Indicator '{name}': Value/Param '{val_or_param}' (resolved to '{val}') is not a valid number or column. Available columns: {list(df.columns)}")
                        continue
                # Compute this step (always a product for your use case)
                try:
                    step_result = None
                    if op == "+":
                        step_result = left + val_operand
                    elif op == "-":
                        step_result = left - val_operand
                    elif op == "*":
                        step_result = left * val_operand
                    elif op == "/":
                        step_result = left / val_operand
                    elif op == "**":
                        step_result = left ** val_operand
                    else:
                        continue
                except Exception:
                    continue
                # Chain with previous result using Combination
                if result is None or prev_comb is None:
                    result = step_result
                else:
                    if prev_comb == "+":
                        result = result + step_result
                    elif prev_comb == "-":
                        result = result - step_result
                    elif prev_comb == "*":
                        result = result * step_result
                    elif prev_comb == "/":
                        result = result / step_result
                    elif prev_comb == "**":
                        result = result ** step_result
                    elif prev_comb == "END":
                        result = step_result  # Or break, but here we just reset
                    else:
                        raise ValueError(f"Unknown combination operator: {prev_comb}")
                prev_comb = comb
            if result is not None:
                df[name] = result
                # Debug: Check if result is all zeros or NaN
                if result.isna().all():
                    print(f"[WARNING] Indicator '{name}' calculated but all values are NaN. Check source columns.")
                elif (result == 0).all():
                    print(f"[WARNING] Indicator '{name}' calculated but all values are 0. Check source columns and parameters.")
            else:
                print(f"[ERROR] Indicator '{name}' could not be calculated - no valid steps were processed.")
    elif builder_df is not None:
        for _, row in builder_df.iterrows():
            name = row.get("Indicator Name")
            ind_a = row.get("Indicator A")
            op = row.get("Operator")
            val_or_param = row.get("Value / Param")
            if not (name and ind_a and op and val_or_param):
                continue
            if str(ind_a) not in df.columns:
                continue
            val = param_map.get(str(val_or_param), val_or_param)
            if str(val) in df.columns:
                val_operand = df[str(val)]
            else:
                try:
                    val_operand = float(val)
                except Exception:
                    continue
            try:
                if op == "+":
                    df[name] = df[str(ind_a)] + val_operand
                elif op == "-":
                    df[name] = df[str(ind_a)] - val_operand
                elif op == "*":
                    df[name] = df[str(ind_a)] * val_operand
                elif op == "/":
                    df[name] = df[str(ind_a)] / val_operand
                elif op == "**":
                    df[name] = df[str(ind_a)] ** val_operand
            except Exception:
                continue
    
    return df

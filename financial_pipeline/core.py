"""Leakage-safe, auditable market-data, forecasting and backtest primitives."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib, json
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

@dataclass(frozen=True)
class Split:
    train: pd.DataFrame; validation: pd.DataFrame; test: pd.DataFrame

def normalise_yfinance_frame(frame: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalise current one-ticker yfinance layouts; reject ambiguous responses."""
    if frame is None or frame.empty: raise ValueError("Market-data response is empty; provider returned no daily bars")
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        first, second = result.columns.get_level_values(0), result.columns.get_level_values(1)
        if set(REQUIRED_COLUMNS).intersection(first): result.columns = first
        elif set(REQUIRED_COLUMNS).intersection(second): result.columns = second
        else: raise ValueError("Unrecognised MultiIndex market-data columns")
    if result.columns.duplicated().any(): raise ValueError("Ambiguous duplicate market-data columns")
    dates = pd.to_datetime(result.index, errors="coerce", utc=True)
    if dates.isna().any(): raise ValueError("Market-data index contains invalid dates")
    result.index = dates.tz_convert(None).normalize(); result.index.name = "Date"
    return result.sort_index()

def validate_market_data(frame: pd.DataFrame, min_rows: int = 100) -> dict:
    """Validate raw unadjusted OHLCV; never silently repair source data."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex): raise ValueError("Market-data index must be timestamps")
    if frame.index.tz is not None: raise ValueError("Market-data timestamps must be timezone-naive dates")
    duplicates = int(frame.index.duplicated().sum())
    if duplicates: raise ValueError(f"Duplicate market dates: {duplicates}")
    if not frame.index.is_monotonic_increasing: raise ValueError("Market dates are not chronological")
    values = frame.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(values.isna().sum().sum())
    if missing_cells: raise ValueError(f"Missing or nonnumeric OHLCV values: {missing_cells}")
    if not np.isfinite(values.to_numpy(float)).all(): raise ValueError("Non-finite OHLCV values detected")
    if len(frame) < min_rows: raise ValueError(f"Only {len(frame)} observations; require at least {min_rows}")
    if (values[["Open","High","Low","Close"]] <= 0).any().any() or (values.Volume < 0).any(): raise ValueError("Non-positive prices or negative volume detected")
    invalid = (values.High < values[["Open","Close","Low"]].max(axis=1)) | (values.Low > values[["Open","Close","High"]].min(axis=1))
    if invalid.any(): raise ValueError("Invalid OHLC consistency: High/Low do not bound Open and Close")
    return {"records":len(frame),"start_date":frame.index.min().date().isoformat(),"end_date":frame.index.max().date().isoformat(),"missing_ohlcv_cells":0,"duplicate_dates":0,"finite_ohlcv":True,"ohlc_consistent":True}

def download_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not ticker or not ticker.replace("-", "").isalnum() or pd.Timestamp(start) >= pd.Timestamp(end): raise ValueError("Ticker must be non-empty and start date must precede end date")
    try:
        import yfinance as yf
        raw = yf.download(ticker.upper(), start=start, end=end, auto_adjust=False, progress=False, threads=False)
    except Exception as error: raise RuntimeError(f"Yahoo Finance download failed for {ticker}: {error}") from error
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {ticker} ({start} to {end}); check ticker, dates, or provider availability")
    data = normalise_yfinance_frame(raw, ticker); validate_market_data(data); return data

def chronological_split(frame: pd.DataFrame, train_ratio: float=.60, validation_ratio: float=.20) -> Split:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio+validation_ratio >= 1: raise ValueError("Split ratios must be positive and leave a test partition")
    a,b=int(len(frame)*train_ratio),int(len(frame)*(train_ratio+validation_ratio)); out=Split(frame.iloc[:a].copy(),frame.iloc[a:b].copy(),frame.iloc[b:].copy())
    if min(map(len,(out.train,out.validation,out.test))) == 0: raise ValueError("A split partition is empty")
    return out

def fit_standardizer(train: pd.DataFrame, columns: Iterable[str]=( "Close",)) -> dict:
    values=train.loc[:,list(columns)].astype(float)
    if not np.isfinite(values.to_numpy()).all(): raise ValueError("Cannot fit standardizer to non-finite data")
    return {c:{"mean":float(values[c].mean()),"std":float(values[c].std(ddof=0) or 1)} for c in values}

def build_lagged_features(close: pd.Series) -> tuple[pd.DataFrame,pd.Series]:
    """At t every feature uses prices at or before t-1."""
    table=pd.DataFrame(index=close.index); table["lag_1"]=close.shift(1); table["mean_5"]=close.shift(1).rolling(5).mean(); table["mean_20"]=close.shift(1).rolling(20).mean(); table["return_1"]=close.pct_change().shift(1); table=table.dropna()
    return table,close.loc[table.index]

def _fit_ridge(x,y,alpha):
    mean,std=x.mean(0),x.std(0); std[std==0]=1; design=np.c_[np.ones(len(x)),(x-mean)/std]; penalty=np.eye(design.shape[1])*alpha; penalty[0,0]=0
    return np.linalg.solve(design.T@design+penalty,design.T@y),(mean,std)
def _predict_ridge(x,coef,scale):
    mean,std=scale; return np.c_[np.ones(len(x)),(x-mean)/std]@coef

def forecast_models(split: Split) -> tuple[pd.DataFrame,pd.DataFrame]:
    full=pd.concat([split.train,split.validation,split.test]); features,target=build_lagged_features(full.Close)
    def part(index): return features.loc[features.index.intersection(index)],target.loc[target.index.intersection(index)]
    xt,yt=part(split.train.index); xv,yv=part(split.validation.index); xs,ys=part(split.test.index)
    if min(len(xt),len(xv),len(xs))<2: raise ValueError("Insufficient lagged observations in a partition")
    choices=(.01,.1,1.,10.); alpha=min(choices,key=lambda a:np.mean((yv.to_numpy()-_predict_ridge(xv.to_numpy(),*_fit_ridge(xt.to_numpy(),yt.to_numpy(),a)))**2))
    coef,scale=_fit_ridge(pd.concat([xt,xv]).to_numpy(),pd.concat([yt,yv]).to_numpy(),alpha)
    predictions=pd.DataFrame({"actual_close":ys,"actual_open":full.loc[xs.index,"Open"],"previous_close":full.Close.shift(1).loc[xs.index],"persistence":xs.lag_1,"moving_average_20":xs.mean_20,"ridge_autoregression":_predict_ridge(xs.to_numpy(),coef,scale)},index=xs.index); predictions.index.name="Date"
    return predictions,pd.DataFrame([{"model":"ridge_autoregression","selected_alpha":alpha,"validation_observations":len(xv),"selection_metric":"validation_mse"}])

def forecasting_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if not {"actual_close","previous_close","persistence"}.issubset(predictions): raise ValueError("Predictions must contain actual_close, previous_close and persistence")
    actual,prior=predictions.actual_close.to_numpy(float),predictions.previous_close.to_numpy(float)
    if not np.isfinite(np.c_[actual,prior]).all() or (actual<=0).any(): raise ValueError("Invalid actual or previous close")
    baseline=np.mean(abs(actual-predictions.persistence.to_numpy(float))); rows=[]
    for model in [c for c in predictions if c not in {"actual_close","actual_open","previous_close"}]:
        pred=predictions[model].to_numpy(float)
        if not np.isfinite(pred).all(): raise ValueError(f"Invalid forecast values for {model}")
        err=actual-pred; mae=float(np.mean(abs(err))); rmse=float(np.sqrt(np.mean(err**2)))
        rows.append({"model":model,"observations":len(actual),"mae":mae,"rmse":rmse,"mape_pct":float(np.mean(abs(err/actual))*100),"nrmse_pct":float(rmse/np.mean(actual)*100),"directional_accuracy":float(np.mean((pred-prior>0)==(actual-prior>0))),"skill_vs_persistence_mae":float(1-mae/baseline) if baseline else 0.,"error_p05":float(np.quantile(err,.05)),"error_p95":float(np.quantile(err,.95))})
    return pd.DataFrame(rows)

def backtest_long_flat(predictions: pd.DataFrame, model: str, cost_bps: float=5.) -> tuple[pd.DataFrame,dict]:
    """Prior-close forecast signal, next available open fill, then open-to-close return."""
    need={"actual_close","actual_open","previous_close",model}
    if cost_bps<0 or not need.issubset(predictions): raise ValueError("Backtest requires nonnegative cost plus open, close, previous close and forecast")
    p=predictions; position=(p[model]>p.previous_close).astype(float); turnover=position.diff().abs().fillna(position); gross=position*(p.actual_close/p.actual_open-1); net=gross-turnover*cost_bps/10000; equity=(1+net).cumprod(); dd=equity/equity.cummax()-1; vol=net.std(ddof=1); sharpe=float(np.sqrt(252)*net.mean()/vol) if pd.notna(vol) and vol else 0.
    ledger=pd.DataFrame({"actual_open":p.actual_open,"actual_close":p.actual_close,"previous_close":p.previous_close,"forecast":p[model],"position":position,"turnover":turnover,"gross_return":gross,"net_return":net,"equity":equity,"drawdown":dd})
    return ledger,{"model":model,"execution_convention":"prior_close_signal_next_open_to_close","total_return_pct":float((equity.iloc[-1]-1)*100),"annualized_sharpe":sharpe,"max_drawdown_pct":float(dd.min()*100),"trade_count":int(turnover.sum()),"cost_bps_per_turnover":cost_bps,"observations":len(ledger)}

def file_sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write_manifest(output_dir: Path, manifest: dict) -> Path:
    hashes={p.name:file_sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name!="manifest.json"}; path=output_dir/"manifest.json"; path.write_text(json.dumps({**manifest,"output_sha256":hashes},indent=2,sort_keys=True)+"\n"); return path
def verify_manifest(path: Path) -> bool:
    """Return false when any manifest-listed output has changed or disappeared."""
    manifest=json.loads(path.read_text())
    return all((path.parent/name).is_file() and file_sha256(path.parent/name)==digest for name,digest in manifest.get("output_sha256",{}).items())
def export_csvs(output_dir: Path, quality: pd.DataFrame, forecasts: pd.DataFrame, strategies: pd.DataFrame, selections: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    for name,frame in {"data_quality_summary.csv":quality,"forecast_metrics.csv":forecasts,"strategy_metrics.csv":strategies,"model_selection.csv":selections}.items(): frame.to_csv(output_dir/name,index=False)

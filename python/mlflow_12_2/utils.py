
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure


def plot_ts_data(
    time_data: pd.Series,
    value_data: pd.Series,
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None,
    lower_ci: Optional[pd.Series] = None, 
    upper_ci: Optional[pd.Series] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    
    ts = pd.Series(value_data.values, index=time_data)
    
    plot_ts(
        ts, 
        start_time,
        end_time,
        lower_ci,
        upper_ci,
        xlabel,
        ylabel,
        title,
    )


def plot_ts(
    ts: pd.Series, 
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None,
    lower_ci: Optional[pd.Series] = None, 
    upper_ci: Optional[pd.Series] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    ts = filter_time_series(ts, start_time, end_time)

    if lower_ci and upper_ci:
        lower_ci = filter_time_series(lower_ci, start_time, end_time)
        upper_ci = filter_time_series(upper_ci, start_time, end_time)
    
    fig, ax = plt.subplots()

    _plot_ts_on_axis(ax, ts, lower_ci, upper_ci)

    _set_plot_properties(
        fig,
        ax,
        add_legend=False, 
        xlabel=xlabel,
        ylabel=ylabel, 
        title=title,
    )

    fig.show()
    
    
def plot_multiple_ts(
    ts_list: list[pd.Series],
    start_time: str = None,
    end_time: str = None,
    lower_ci_list: Optional[list[pd.Series]] = None,
    upper_ci_list: Optional[list[pd.Series]] = None,
    labels: Optional[list[str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot multiple time series predictions with their respective confidence 
    intervals on the same plot.
    """
    labels = _prepare_labels(len(ts_list), labels)
    
    fig, ax = plt.subplots()
    
    for i in range(len(ts_list)):
        ts = ts_list[i]
        lower_ci = lower_ci_list[i]
        upper_ci = upper_ci_list[i]
        
        ts = filter_time_series(ts, start_time, end_time)

        if lower_ci is not None and upper_ci is not None:
            lower_ci = filter_time_series(lower_ci, start_time, end_time)
            upper_ci = filter_time_series(upper_ci, start_time, end_time)
            
        color = _get_color(i)
        label = labels[i]
        
        _plot_ts_on_axis(ax, ts, lower_ci, upper_ci, color, label)

    _set_plot_properties(
        fig, 
        ax, 
        True, 
        xlabel=xlabel, 
        ylabel=ylabel, 
        title=title,
    )
    fig.show()   


def filter_time_series(
    ts: pd.Series, 
    start_time: Optional[str] = None, 
    end_time: Optional[str] = None
) -> pd.Series:
    if start_time is not None:
        ts = ts[ts.index >= start_time]
    if end_time is not None:
        ts = ts[ts.index <= end_time]
    return ts 


def read_csv_series(path: str) -> pd.Series:
    df = pd.read_csv(path) 
    series = pd.Series(df["0"].values, index=df.date)
    return series 


def sm_arima_predict_w_interval(
    model, 
    ts: pd.Series, 
    alpha: float = 0.05
) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    return _sm_predict_w_interval(
        model, 
        ts,
        alpha,
        "mean",
        "mean_ci_lower",
        "mean_ci_upper",
    )


def sm_ets_predict_w_interval(
    model, 
    ts: pd.Series, 
    alpha: float = 0.05
) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    return _sm_predict_w_interval(
        model, 
        ts,
        alpha,
        "mean_numerical",
        "pi_lower",
        "pi_upper",
    )


def prophet_predict_w_interval(
    model,
    ts: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    future = (
        pd.DataFrame({
            'ds': ts.index.tolist(),
        })
        .reset_index(drop=True)
    )

    forecast = model.predict(future)

    ts_h = pd.Series(
        forecast['yhat'].values, 
        index=forecast['ds'],
    )

    ts_h_lower = pd.Series(
        forecast['yhat_lower'].values, 
        index=forecast['ds'],
    )

    ts_h_upper = pd.Series(
        forecast['yhat_upper'].values, 
        index=forecast['ds'],
    )

    return ts_h, ts_h_lower, ts_h_upper


def join_forecast_to_series(
    ts: pd.Series,
    ts_index: pd.Index,
    ts_h: pd.Series,
    ts_h_lower: pd.Series,
    ts_h_upper: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    ts_with_h = ts.copy()
    # Recovering the series index
    ts_with_h.index = ts_index
    ts_with_h = pd.concat([ts_with_h, ts_h])

    ts_with_h_lower = ts_with_h.copy()
    ts_with_h_lower[ts.index] = None
    ts_with_h_lower[ts_h_lower.index] = ts_h_lower

    ts_with_h_upper = ts_with_h.copy()
    ts_with_h_upper[ts.index] = None
    ts_with_h_upper[ts_h_upper.index] = ts_h_upper
    
    return ts_with_h, ts_with_h_lower, ts_with_h_upper


def _sm_predict_w_interval(
    model, 
    ts: pd.Series, 
    alpha: float = 0.05,
    name_mean: str = "mean",
    name_ci_lower: str = "mean_ci_lower",
    name_ci_upper: str = "mean_ci_upper",

) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    pred = model.get_prediction(
        start=ts.index[0], 
        end=ts.index[-1],
    )

    df_pred = pred.summary_frame(alpha=alpha)

    ts_h = pd.Series(
        df_pred[name_mean].values, 
        index=df_pred.index,
    )
    ts_h.name = "predicted_mean"

    ts_h_lower = pd.Series(
        df_pred[name_ci_lower].values, 
        index=df_pred.index,
    )
    ts_h_lower.name = "predicted_mean_ci_lower"

    ts_h_upper = pd.Series(
        df_pred[name_ci_upper].values, 
        index=df_pred.index,
    )
    ts_h_upper.name = "predicted_mean_ci_upper"

    return ts_h, ts_h_lower, ts_h_upper


def _set_plot_properties(
    fig: figure.Figure,
    ax: plt.Axes,
    add_legend: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Set properties for the plot.
    """
    ax.set_xticks(ax.get_xticks()[1:], ax.get_xticklabels()[1:], rotation=70)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if add_legend:
        fig.legend(loc='upper left')
    
    
def _plot_ts_on_axis(
    ax: plt.Axes, 
    ts: pd.Series,
    lower_ci: Optional[pd.Series] = None, 
    upper_ci: Optional[pd.Series] = None,
    color: str = "#1f77b4", 
    label: Optional[str] = None,
):
    """
    Plot a time series and CI on an axis with a given color and label.
    """
    ax.plot(ts.index, ts.values, color=color, label=label)
    
    if lower_ci is not None and upper_ci is not None:
        ax.fill_between(
            ts.index, 
            lower_ci, 
            upper_ci, 
            alpha=0.2, 
            color=color, 
            label=f'{label} Confidence Interval'
        )

        
def _get_color(index: int) -> str:
    """
    Get a color based on an index.
    """
    colors = ['r', 'g', 'b', 'k'] # Add more colors as needed
    return colors[index % len(colors)]


def _prepare_labels(n_ts: int, labels: Optional[list[str]]) -> list[str]:
    """
    Create labels if they are None.
    """
    if labels:
        return labels
    
    labels = list()
    for i in range(n_ts):
        labels.append(f'TS {i+1}')
        
    return labels

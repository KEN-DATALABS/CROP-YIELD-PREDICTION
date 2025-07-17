import streamlit as st
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Union



def pie_plot(values: Union[List[float], List[int]],labels: List[str],title: str = "Pie Chart",figsize: Tuple[int, int] = (6, 6)):
    """
    Plots a pie chart using the given values and labels, and optionally displays it in Streamlit.
    Parameters:
    - values: List of numerical data for the pie slices.
    - labels: List of labels corresponding to each pie slice.
    - title: Title of the pie chart (default is "Pie Chart").
    - ax: Optional matplotlib axis to plot on. If None, a new figure is created.
    - figsize: Tuple indicating figure size (default is (6, 6)).
    """
    # Create figure and axis if not provided
    fig, ax = plt.subplots(figsize=figsize)
    # Slightly explode all slices for emphasis
    explode = [0.05] * len(values)
    # Draw the pie chart
    ax.pie(values,labels=labels,autopct='%1.1f%%',startangle=140,explode=explode,textprops={'fontsize': 14})
    # Set title
    ax.set_title(title, fontsize=16)
    # Render in Streamlit
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to free memory


def bar_plot(df,x: str,y: str,title: str = "Bar Plot",xlabel: Optional[str] = None,ylabel: Optional[str] = None,
             hue: Optional[str] = None,palette: str = "viridis",figsize: Tuple[int, int] = (10, 6),rotation_x: int = 45,
             rotation_y: int = 0, legend_title=None, legend_loc=None, legend_bbox=None):
    """
    Plots a customizable bar plot using seaborn and matplotlib, and displays it in Streamlit.
    Parameters:
    - df: Pandas DataFrame containing the data.
    - x: Column name for the x-axis.
    - y: Column name for the y-axis.
    - title: Title of the plot (default: 'Bar Plot').
    - xlabel: Label for the x-axis (default: same as `x`).
    - ylabel: Label for the y-axis (default: same as `y`).
    - hue: Optional column name for grouping by color.
    - palette: Color palette to use (default: 'viridis').
    - figsize: Size of the plot (default: (10, 6)).
    - rotation_x: Rotation angle for x-axis ticks (default: 80).
    - rotation_y: Rotation angle for y-axis ticks (default: 0).
    """
    # Set default labels if not provided
    xlabel = xlabel or x
    ylabel = ylabel or y
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df,x=x,y=y,hue=hue,palette=palette,ax=ax)
    # Style axis ticks
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation_x)
        tick.set_fontsize(11)
    for tick in ax.get_yticklabels():
        tick.set_rotation(rotation_y)
        tick.set_fontsize(11)
    # Legend handling
    if hue and legend_title:
        ax.legend(title=legend_title, title_fontsize=12, fontsize=10, loc=legend_loc, bbox_to_anchor=legend_bbox)
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    # Adjust layout and show in Streamlit
    plt.tight_layout()
    st.pyplot(fig)

def count_plot(df: pd.DataFrame, column: str = 'Area', xlabel: Optional[str] = None, ylabel: Optional[str] = None, 
               title: str = "Count Plot", palette: str = 'viridis', figsize: Tuple[int, int] = (10, 6), rotation_x: int = 0,
               rotation_y: int = 0):
    """
    Displays a horizontal count plot for a specified categorical column in a DataFrame.
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column (str): The categorical column to count and plot (default: 'Area').
    - xlabel (str, optional): Custom label for the x-axis.
    - ylabel (str, optional): Custom label for the y-axis.
    - title (str): Title of the plot.
    - palette (str): Seaborn color palette.
    - figsize (tuple): Size of the plot.
    - rotation_x (int): Rotation angle for x-axis labels.
    - rotation_y (int): Rotation angle for y-axis labels.
    """
    # Validate input column
    xlabel = xlabel or "Frequency"
    ylabel = ylabel or column

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, y=column, palette=palette, ax=ax)
    # Axis styling
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation_x)
        tick.set_fontsize(12)
    for tick in ax.get_yticklabels():
        tick.set_rotation(rotation_y)
        tick.set_fontsize(10)
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def line_plot(data: Union[Series, DataFrame], x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None,
              title: str = "Line Plot", xlabel: Optional[str] = None, ylabel: Optional[str] = None, color: str = "orange",
              marker: str = "o", linewidth: float = 2.0, figsize: Tuple[int, int] = (10, 6), grid: bool = True):
    """
    A flexible line plot function for Streamlit. Supports both Series and DataFrame input.
    Parameters:
    - data: pandas Series or DataFrame
    - x, y: Column names (for DataFrame)
    - hue: Optional column for color grouping (e.g., Crop)
    - title, xlabel, ylabel: Plot labels
    - color: Line color for Series plots or when no hue
    - marker: Marker style
    - linewidth: Line thickness
    - figsize: Size of figure
    - grid: Show gridlines
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Plotting logic
    if isinstance(data, Series):
        data.plot(ax=ax, color=color, linewidth=linewidth, marker=marker)
        if xlabel is None: xlabel = data.index.name or "Index"
        if ylabel is None: ylabel = "Value"
    else:
        sns.lineplot(data=data, x=x, y=y, hue=hue, marker=marker, linewidth=linewidth,
                     ax=ax, palette="tab10" if hue else None, color=None if hue else color)
        if xlabel is None: xlabel = x
        if ylabel is None: ylabel = y
    # Styling
    ax.set_title(title, fontsize=17)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def scatter_plot(df, x: str, y: str, title: str = "Scatter Plot",
                xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                hue: Optional[str] = None, alpha: float = 0.6,
                figsize: Tuple[int, int] = (10, 6), legend_title=None,
                legend_loc='upper right', legend_bbox=(1.05, 1)):
    """
    Reusable scatter plot function for Streamlit using seaborn & matplotlib.
    Parameters:
    - df: Pandas DataFrame
    - x: Column for x-axis
    - y: Column for y-axis
    - title: Title of the plot
    - xlabel, ylabel: Axis labels
    - hue: Column name for color grouping
    - alpha: Transparency of dots
    - figsize: Tuple for figure size
    - legend_title: Title for the legend
    - legend_loc: Legend location
    - legend_bbox: BBox anchor for legend positioning
    """
    xlabel = xlabel or x
    ylabel = ylabel or y

    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=alpha, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    for label in ax.get_xticklabels():
        label.set_fontsize(13)
    for label in ax.get_yticklabels():
        label.set_fontsize(13)
    if hue:
        ax.legend(title=legend_title or hue, bbox_to_anchor=legend_bbox, loc=legend_loc, fontsize=12, title_fontsize=12)
    st.pyplot(fig)

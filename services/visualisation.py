import streamlit as st
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_and_close(fig):
    st.pyplot(fig)
    plt.close(fig)

def style_axis_ticks(ax, axis='x', rotation=None, fontsize=12):
    if axis == 'x':
        for tick in ax.get_xticklabels():
            if rotation is not None: 
                tick.set_rotation(rotation)
            tick.set_fontsize(fontsize)
    elif axis == 'y':
        for tick in ax.get_yticklabels():
            if rotation is not None:
                tick.set_rotation(rotation)
            tick.set_fontsize(fontsize)

# Pie chart plot function
def plot_pie_chart(values: list, labels: list, title: str, colors=plt.cm.Paired.colors, ax=None):
    '''
    Values: List of integers
    labels: List of string for each element in the value
    title: Title of the the chart/plot
    color: Color palette to be used
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        ax.pie(values, labels=labels, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}, 
            textprops={'fontsize': 11, 'weight': 'roman'}, 
            colors=colors, 
            autopct='%1.2f%%')
        ax.set_title(title, fontsize=15, color='darkblue')
    except Exception as e:
        print(f"Error in plot_pie_chart: {e}")

# Doughnut chart plot function
def plot_doughnut_chart(values: list, labels: list, title: str, colors=sns.color_palette("muted"), ax=None):
    '''
    Values: List of integers
    labels: List of string for each element in the value
    title: Title of the the chart/plot
    color: Color palette to be used
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        ax.pie(values, labels=labels, 
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'width': 0.3}, 
            textprops={'fontsize': 11, 'weight': 'bold'}, 
            colors=colors, 
            startangle=90, 
            autopct='%1.2f%%')
        ax.set_title(title, fontsize=15, color='darkblue')
    except Exception as e:
        print(f"Error in plot_doughnut_chart: {e}")

# Bar chart plot function
def plot_bar_chart(data: DataFrame, x: str, y: str, title: str, hue:str=None, ax=None):
    '''
    x: List of integers on x axies
    y: List of integers on y axies
    title: Title of the the chart/plot
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        ax.bar(
            data=data,
            x=x, height=y,
            color=plt.cm.Paired(range(len(x))), 
            edgecolor='black', 
            hue=hue,
            linewidth=0.8)
        ax.set_title(title, fontsize=15, color='darkblue')
        ax.set_xticklabels(x, rotation=60, fontsize=11)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        for index, value in enumerate(y):
            ax.text(x=index, y=value + 0.02 * max(y), s=f'{str(round(value, 3))}', ha='center')
    except Exception as e:
        print(f"Error in plot_bar_chart: {e}")

# Count plot function
def plot_count_plot(data: DataFrame, x: str, title: str, hue=None, ax=None):
    '''
    data: Pandas dataframe for source data
    x: string values for x axies
    hue: Grouping class column
    title: Title of the the chart/plot
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        sns.countplot(data=data, x=x, hue=hue, palette='Set2', ax=ax)
        ax.set_title(title, fontsize=16, color='darkblue')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=11, color='#34495e')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        for container in ax.containers:
            ax.bar_label(container)
    except Exception as e:
        print(f"Error in plot_count_plot: {e}")

# Scatter plot function
def plot_scatter_plot(x: list, y: list, title: str, ax=None):
    '''
    x: List of integers on x axies
    y: List of integers on y axies
    title: Title of the the chart/plot
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        ax.scatter(x=x, y=y,
            color=plt.cm.Paired(range(len(x))), 
            edgecolor='black', 
            linewidth=0.8)
        ax.set_title(title, fontsize=15, color='darkblue')
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=60, fontsize=11)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        for index, value in enumerate(y):
            ax.text(x=index, y=value + 0.02 * max(y), s=f'{str(round(value, 3))}', ha='center')
    except Exception as e:
        print(f"Error in plot_scatter_plot: {e}")

# Violin plot function
def plot_violin_plot(data: DataFrame, x: str, title: str, hue=None, ax=None):
    '''
    data: Pandas dataframe for source data
    x: string values for x axies
    hue: Grouping class column
    title: Title of the the chart/plot
    ax: Axes 
    '''
    try:
        if ax is None:
            ax = plt.gca()
        sns.violinplot(data=data, x=x, hue=hue, palette='Set2', ax=ax)
        ax.set_title(title, fontsize=16, color='darkblue')
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    except Exception as e:
        print(f"Error in plot_violin_plot: {e}")



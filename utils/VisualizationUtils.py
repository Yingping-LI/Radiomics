#!/usr/bin/env python
# coding: utf-8

## For plots
import seaborn as sns
import matplotlib.pyplot as plt

def get_color_list():
    color_list = list({
       'lightskyblue':'#87CEFA',
       'tomato':'#FF6347',
       'gray': '#808080',
       'purple': '#800080',
       'red': '#FF0000',
       'palegreen':'#98FB98',
       'lime': '#00FF00',
       'darkred': '#8B0000',
       'orchid':'#DA70D6',
       'brown':'#A52A2A',
       'crimson':'#DC143C',
       'olive':  '#808000',
       'blueviolet': '#8A2BE2',
       'yellow':'#FFFF00',  
       'green': '#008000',
       }.values())
    
    return color_list

def change_bar_width(ax, new_width) :
    """
    Change the bar width.
    """
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_width
        patch.set_width(new_width)
        patch.set_x(patch.get_x() + diff * .5)
        
def plot_crosstab(cross_table, stacked, save_plot_path):
    """
    Plot cross table results.
    """
        
    # plot the cross table.
    ax=cross_table.plot.bar(stacked=stacked, xlabel="") 
    plt.xticks(rotation=360)
    plt.ylabel("Number of Patient")
    plt.grid(True)
    plt.legend(loc="best")
    
    #change bar width
    if stacked:
        change_bar_width(ax, new_width=0.4)
    
    # add text! Patches is everything inside of the chart.
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        label_text = int(height)  # f'{height:.2f}' to format decimal values

        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)   
        
    # save plots   
    plt.savefig(save_plot_path)
    plt.show()
#!/usr/bin/env python
# coding: utf-8

# In[16]:


# !pip install pandas
# !pip install matplotlib
# !pip install seaborn
# !pip install pip install scikit-learn
# !pip install plotly
# !pip install openpyxl
get_ipython().system('pip install xlrd')


# # Data Visualization 101 - part II : The Notebook
# advanced visualization functions

# ### import packages:

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ### Declaring constant variables:

# In[4]:


BINN_NUM = 6 # to create age groups at the Heart Disease dataset
TITANIC = 'titanic3.xls' # the path of the titatic dataset file
HEART = 'heart.csv' # the path of the heart disease dataset file
AIRBNB = 'listings.csv' # the path of the Airbnb Amsterdam Listing dataset file
RANDOMNESS, SAMPLES_NUM, STRING = 42, 1000, "ABCDEFGHIJ" # exemplary dataset for the 1st JoyPlot


# ### creating and dataframes:

# #### (I) Boston Housing dataset:

# In[7]:


from sklearn.datasets import load_boston


# In[8]:


boston_data = load_boston()
print(boston_data.DESCR)


# In[9]:


x_bos = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
y_bos = pd.Series(boston_data.target, name="PRICE")
boston = x_bos.join(y_bos)
display(boston.sample(5))


# #### (II) Tips dataset:

# In[11]:


import plotly.express as px


# In[12]:


tips = px.data.tips()
target = "tip"
features = [col for col in tips.columns if col != target]
x_tips, y_tips = tips[features], tips[target]
tips = x_tips.join(y_tips)
tips.sample(5)


# #### (III) Titanic dataset:

# In[17]:


titanic = pd.read_excel(TITANIC).drop_duplicates()
titanic.sample(5)


# #### (IV) Heart Disease dataset:

# In[18]:


heart = pd.read_csv(HEART)

age_cutt_off = np.arange(heart.age.min(), heart.age.max()+1, (heart.age.max() - heart.age.min())/ BINN_NUM).tolist()
heart['age_binned'] = pd.cut(heart['age'], age_cutt_off) # creates age intervals ((min, max])
# heart["sex_male"] = pd.Series([int(not c) for c in heart.sex])

display(heart.sample(5))


# #### (V) Airbnb Amsterdam Listing dataset:

# In[19]:


listings = pd.read_csv(AIRBNB)

# removing some features: just so ot would be easier on the eyes
listings = listings[['property_type', 'room_type', 'bed_type', 'beds', 'accommodates', 
                     'review_scores_rating', 'number_of_reviews',
                     'review_scores_value', 'review_scores_accuracy', 'price']] 

listings = listings.drop_duplicates()
listings.dropna(inplace=True)
listings['price'] = listings['price'].str.extract('(\d+)')
listings['price (log)'] = listings['price'].apply(lambda x: np.log(int(x)))
listings.sample(5)


# #### (VI) Exemplary dataset 
# (to show a distribution pattern - for the 1st joyplot example):

# In[20]:


rand_sample = np.random.RandomState(RANDOMNESS)
x = rand_sample.randn(SAMPLES_NUM)
letter = np.tile(list(STRING), int(SAMPLES_NUM/len(STRING)))
patterned_data = pd.DataFrame(dict(x=x, letter=letter))
ord_n = patterned_data.letter.map(ord)
patterned_data["x"] += ord_n
patterned_data.sample(5)


# ## Ploting:

# ### 1. Fast Hacks with DataPrep EDA and Profiling libraries
# The following 2 packages are useful mainly for fast EDA, but they can also be a great plotting tool option. \
# Each of the modules creates a dynamic and interactive data presentation.
# 
# 
# They use graphs to present different aspects in the given data but they also present additional information (like missing values, \
# duplications, etc') - meaning they aren't a direct visualization tool.

# #### - Pandas Profiling
# As an example, we will use the pandas_profiling ProfileReport function the the heart disease dataset.

# In[21]:


# !pip install pandas-profiling


# In[22]:


from pandas_profiling import ProfileReport


# In[23]:


profile = ProfileReport(heart, title="Pandas Profiling Report")
profile.to_widgets()


# _For more info regarding each section [click here](https://towardsdatascience.com/the-best-exploratory-data-analysis-with-pandas-profiling-e85b4d514583)_ 
# 
# 
# 
# _**NOTE:** The main disadvantage of pandas profiling is its use with large datasets. With the increase in the size of the data the time to generate the report also increases a lot. But there is a way around it like you can read  [here](https://towardsdatascience.com/exploratory-data-analysis-with-pandas-profiling-de3aae2ddff3) and [here](https://medium.com/analytics-vidhya/pandas-profiling-5ecd0b977ecd)._

# #### - dataprep

# _**NOTE:** Pandas Profiling helps with data Profiling, and data profiling and EDA aren't the same. If you are looking for an additional library to help with your EDA you should look over the DataPrep.eda package which might be beneficial to you. \
# If you like to read more about the difference between the two packages please check out [this medium article](https://towardsdatascience.com/exploratory-data-analysis-dataprep-eda-vs-pandas-profiling-7137683fe47f) that covers just that- Enjoy!_

# For this example, we will use the heart disease dataset yet again.

# In[24]:


# !pip install dataprep


# In[25]:


from dataprep.eda import plot


# In[12]:


plot(heart)


# _**NOTE: For more info about the DataPrep library you can check [this medium blog](https://towardsdatascience.com/dataprep-eda-accelerate-your-eda-eb845a4088bc)._

# ## Categorical & Numerical features combination plots:
# ### 2. JoyPlot (Ridgeline plot)
# Joyplot (aka Ridgeline plot or Overlapping densities plot) shows the distribution of a numeric value of several groups, \
# each group per row. The groups can be divided by the values of a categorical feature. \
# Although the distribution can be presented as a histogram, it is usually portrayed by a KDE plot. All rows share x-axis values, \
# meaning we should consider scaling all groups data accordingly. 
# 
# 
# It is recommend to use Joyplots where there are many groups but not too many (- let's say that the rule of thumb is \
# somewhere between 5 to 12 groups). Joyplot major advantage is when we want to compare and to check for a distribution pattern \
# between several groups. So when there is a clear pattern, the plot is very useful, otherly, it will give us less insights.

# #### (I) A Simple JoyPlot
# *based on seaborn joyplot documentation*

# In[50]:


def joy_plot(df, x_axis, y_axis, title=None):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # creating a sequential palette from the cubehelix system:
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7) 
    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row=y_axis, hue=y_axis, # hue assigns the colors according to the row variable
                      aspect=13, height=.6, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, x_axis, clip_on=False, shade=True, alpha=1, 
          lw=1.5, bw=.2) #  apply the plotting function sns.kdeplot to the x_axis variable
    # now g has the y_axis as the row. Each row will plot the distribution of the x_axis variable

    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    g.map(
        lambda x, color, label: plt.gca().text(0, .2, label, fontweight="bold", 
                                                 color=color, ha="left", va="center", 
                                                 transform=plt.gca().transAxes), 
          str(x_axis)
    )

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=0.05)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    if title:
        plt.suptitle(title, fontsize=16)


# The JoyPlot rows usually overlap a bit, but eventually it is a style option designed to benefit us. \
# You may adjust it as you please by changing the "hspace" parameter value.
# 
# 
# Let's look at the following example that uses the Airbnb Amsterdam listing dataset. Note that some of the data were slightly altered, \
# for example: the price column where scaled by log.

# In[51]:


joy_plot(listings, "price (log)", 'bed_type', 
         title="Distribution of bed type to room price (in log)")

plt.show()


# From the Airbnb Amsterdam listing plot we can conclude that, for some reason, Airbeds are sometimes associated with higher priced venues. \
# Furthermore, Couch has quite a narrow bimodal distribution, compared to other values that have a multimodal distribution. \
# That means that for most parts at price ranges of 3.5–4 and 4.5–5 there will be a couch.
# 
# 
# This example is not the best to showcase a patterned distribution between groups. \
# Let's try again and use a dummy-dataset I created to see how we can easily identify a pattern between different groups distribution:

# In[52]:


joy_plot(patterned_data, "x", 'letter', 
         title="Using special dataset to showcase trend distribution of different groups".title())
plt.show()


# From that joyplot above we can see how the value of x is greater for a letter in a higher alphabet position. \
# That is an example of a trend that a joyplot may provide.

# #### (II) Comparison JoyPlot
# The next Joyplot we'll be reffering as Comparison Joyplot. The "Comparison" part is added because we want to diplay per each row \
# the distributions of multiple groups. The groups are the actually values of another categorical feature. \
# For example, we used the heart disease dataset:

# In[53]:


import matplotlib.patches as mpatches


# In[56]:


def joy_plot_for_2(df, x, y, hue, title=None, colors=None):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})        

    if not colors or len(colors) < 2:
        colors = ['powderblue', 'cadetblue']

    # Initialize a FacetGrid object
    g = sns.FacetGrid(data=df, row=x, aspect=6, height=1.8)

    # Drawing the KDEs
    g.map_dataframe(sns.kdeplot, y,
                    bw_adjust=.5, clip_on=False, fill=True, shade=True, alpha=1, linewidth=1.5, 
                    hue=hue, hue_order=[0, 1], palette=colors, multiple='stack')
    g.map(plt.axhline, y=0, lw=2, clip_on=False, color='black')

    # Defining a label for each row in the joyplot
    g.map(
            lambda x, color: plt.gca().text(0, .2, f"ages {str(x.iloc[0])[1:3]} to {str(x.iloc[0])[7:9]}", 
                                            fontweight="bold", color='black', ha="left", va="center", 
                                            transform=plt.gca().transAxes), 
              x
        )

    # Seting all subplots to overlap a bit to create the figure nice and tight
    g.fig.subplots_adjust(hspace=-.25)

    # Removing axes details that we don't need
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)
    
    if title:
        plt.suptitle(title, fontsize=20)
    
    # in order to add the labels we will use mpatches:
    first_patch = mpatches.Patch(color=colors[0], label='Male')
    sec_patch = mpatches.Patch(color=colors[-1], label='Female')
    g.add_legend(facecolor='w', handles=[first_patch, sec_patch], fontsize="x-large")


# In order to display the labels correctly, It seems to work best when I use matplotlib.patches Patch object. \
# Also, AS seen earlier, the "age_binned" is a feature I added, which is based on the original "age" column.

# In[57]:


joy_plot_for_2(heart, "age_binned", 'chol', 'sex', title="Distribution of age to blood cholesterol levels by sex")

plt.show()


# For the information above, it seems that at the youngest age group, females tend to have higher blood cholesterol levels compared to male.

# #### Comparison JoyPlot coupled with count plot (%)
# A different approach to create a joyplot, only now, instead of creating a seaborn FacetGrid object, \
# we will use matplotlib gridspec. This object allows us to place subplots within a figure in a grid layout based on our likings.
# 
# This object is good for us because it also enables us to control the width of each row and column in the figuere \
# and the grid areas per each axes.

# In[58]:


import matplotlib.gridspec as gridspec


# In[59]:


def joy_plot_advanced(df, x, y, hue, group_num, 
                      hue_legend = None,
                      figsize=(12,7), 
                      title=None,
                      with_total_precentages=True,
                      colors=None):
    if not colors or len(colors) < 3:
        colors = ['steelblue', 'cadetblue', 'slategray']
        
    # 'freq' is the total percentage of that age groups out of all groups
    freq = ((df[x].value_counts(normalize = True)\
             .reset_index().sort_values(by = 'index')[x])*100).tolist()

    # Manipulate each axes object on the left (where the joy plot).
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=group_num, 
                           ncols=2, 
                           figure=fig, 
                           width_ratios=[3, 1],
                           height_ratios= [1]*group_num,
                           wspace=0.2, hspace=0.05
                          )
    ax = [None] * (group_num + 1)
    # this function is suited for a specific x values (groups of intervals) 
    features = sorted(df[x].dropna().unique(), key= lambda x: int(str(x)[1:3].strip(".")))
    features_names = [f"{str(gr)[1:3]} - {str(gr)[7:9]}" for gr in features]
    
    for i in range(group_num):
        ax[i] = fig.add_subplot(gs[i, 0]); ax[i]
        ax[i].set_yticks([]); ax[i].spines['left'].set_visible(False)
        ax[i].spines['right'].set_visible(False); ax[i].spines['top'].set_visible(False)        
        ax[i].spines['bottom'].set_edgecolor('#444444'); ax[i].spines['bottom'].set_linewidth(2)
        ax[i].text(0.02, 0.05, 'Age: ' + str(features_names[i]), fontsize=14, transform = ax[i].transAxes) 
        sns.kdeplot(data=df[(df[hue] == 0) & (df[x] == features[i])][y], 
                    ax=ax[i], shade=True, color=colors[0], legend=False)
        sns.kdeplot(data=df[(df[hue] == 1) & (df[x] == features[i])][y], 
                    ax=ax[i], shade=True, color=colors[1], legend=False)
        if i < (group_num - 1): 
            ax[i].set_xticks([]); ax[i].set_xlabel("")
            
    if hue_legend:
        ax[0].legend(hue_legend, facecolor='w')
        
    # adding legends only on the top axes object     
    ax[group_num] = fig.add_subplot(gs[:, 1])
    ax[group_num].spines['right'].set_visible(False); ax[group_num].spines['top'].set_visible(False)
    
    # Manipulate each axes object on the right (where the bar plot).
    ax[group_num].barh(features_names, freq, color=colors[2], height=0.4)
    if with_total_precentages:
        total = len(df[x])
        for p in ax[group_num].patches:
                percentage = '{:.1f}%'.format(100 * p.get_width() / total)
                a = p.get_x() + p.get_width() + 5.0
                b = p.get_y() + (p.get_height() / 2)
                ax[group_num].annotate(percentage, (a, b))
    
    ax[group_num].set_xlim(0,100); ax[group_num].invert_yaxis()
    ax[group_num].text(1.09, -0.04, '(%)', fontsize=10, transform = ax[group_num].transAxes)   
    ax[group_num].tick_params(axis='y', labelsize = 14)
    
    if title:
        plt.suptitle(title, fontsize=16)
        
    plt.tight_layout()


# We;'' use the heart disease dataset with the engineered feature "age_binned" yet again.

# In[60]:


joy_plot_advanced(heart,"age_binned", 'thalach', 'sex', BINN_NUM, 
                  hue_legend = ['Male', 'Female'], 
                  title= "Age to Maximum Achieved Heart Rate distribution by Sex")

plt.show()


# From the left side of the graph above, we can conclude that males tend to have higher values of maximal achieved heart rate than \
# females, except in age groups of 37–45 and 45–53 where it is the opposite. \
# On the right side of the graph we can see the distribution of patients in different age groups.
# 
# 
# Up until now, we saw 3 types of costume-made Joyplots. It should be noted that there are other ways to create joyplots while \
# using certain modules. One of them is [joypy](https://pypi.org/project/joypy/), which I will not cover today.

# ### 3. Multi-featured plots of both categorical and numerical data

# #### (I) Two categorical features & one numerical feature
# 
# There are different ways to showcase distribution of a categorical feature by a numeric feature beside a bar chart or a pie chart. \
# Moreover, most of them can give us more insight about the data in hand. We would focus on 3 of them:

# - **Box plot** is a chart that visually demonstrates the distribution of data while determining it's skewness (we can assess un/symmetry) \
# by the display of the mean and quartiles. This type of chart is an extremely helpful way to visually detect outliers. \
# It can help us to showcase the distribution of a numerical feature across a variety of values of another categorical feature, \
# just like we will see in the following example. \
# Each box plot contains 5 essential elements: minimum and maximum scores, the 1st and the 3rd quartiles and the mean. \ The whiskers are the ranges between the minimum to the 1st quartile and between the 3rd quartile to the maximum. \
# The area between the 1st and the 3rd quintiles, which represents 50% of the main scores, is called Interquartile Range (or IQR in short). \
# All points outside of the 2 ranges are considered as outliers.
# 
# 
# - **Violin plot** is kind of an advanced box plot - it is very similar to a box plot but it also contains a rotated \
# Kernel Density Estimation plot on each side of the "bar", giving it a violin-shape. \
# The addition of up-to-2 KDE plot (on for each side of the violin), set the violin plot to be more suited to display \
# extra dimensionality of the data compared to the common box plot.\
# Furthermore, the ability to present the distribution of all the data, gives the reader additional information, which in turn, \
# can be used to display multimodal distribution (more than the traditional one-peak distribution).
# 
# 
# 
# - **Strip plots** (aka Dot plots) are a bit different from the 2 previous types of graphs, due to the fact that they are, \
# actually a scatter plots. This type of technique is useful in summarizing a small univariate data. \
# Each strip is constructed from multiple data points, which are stacked one-on-top-of-the-other. \
# Each point is located based on the corresponding area in the y-axis, which allows one to see the shape of the data distribution. \
# Overall a stripplot can be helpful to review data distribution and to detect outliers. \
# It is possible to plot a strip plot for every value of a categorical feature, so we will get a side-by-side strip plot, \
# where the x-axis represents the categorical feature.

# _**NOTE:** we will use stripplot when we have a medium number of observations (it's scale it nicely and 'jitter' helps), \
# but if it's not the case, we should use swarmplot, which looks better because the points are less likely to overlap (represantation wise)._

# In[61]:


def cat_num_plots(df, cat_col1: str, cat_col2: str,
                  num_col1: str = None,
                  plots_kinds: list = ['strip', 'box', 'violin'],
                  titles: list = None,
                  major_title: str = None,
                  palette: str = "Set3",
                  style: str = "whitegrid", 
                  figsize: tuple = (12, 4)):
    if style:
        sns.set_theme(style=style)
    sns.set_palette(palette)
    all_kinds = {'strip': sns.stripplot, 'swarm': sns.swarmplot, 
             'box': sns.boxplot, 'violin': sns.violinplot, 'boxen': sns.boxenplot}
    kinds = [value for key, value in all_kinds.items() if key in plots_kinds]
    
    fig, axs = plt.subplots(1, len(kinds), sharey=True, figsize=figsize) 

    for i, ax in enumerate(axs.flat):
        ax = kinds[i](data=df, x=cat_col1, y=num_col1, hue=cat_col2, ax=ax)
        if titles:
            ax.set_title(titles[i], fontsize=12)
    if major_title:
        fig.suptitle(major_title, y=1.08, fontsize=17)
    fig.tight_layout()


# To display the 3 plot types, we used the features "time", "day" and "total_bill" of the tips dataset.

# In[62]:


sns.set_theme(style="white")

cat_num_plots(tips, "day", "time", "total_bill",
             titles=["Strip Plot", "Box Plot", "Violin Plot"],
             major_title='3 features plots: 2 categorical and 1 numerical',
             style=None)

plt.show()


# Because the plots share a y-axis, it might be more readable to add grid lines, like this:

# In[63]:


cat_num_plots(tips, "day", "time", "total_bill",
             titles=["Strip Plot", "Box Plot", "Violin Plot"],
             major_title='3 features plots: 2 categorical and 1 numerical')

plt.show()


# 
# _**NOTE:** we will use stripplot when we have a medium number of observations (it's scale it nicely and 'jitter' helps), \
# but if it's not the case, we should use swarmplot, which looks better because the points are less likely to overlap (represantation wise)._
# 
# Let's see for ourselves how the swarmplot acts compared with the other 3 plot types:

# In[64]:


sns.set_theme(style="white")

cat_num_plots(tips, "day", "time", "total_bill",
              plots_kinds=['swarm', 'strip', 'boxen', 'box'],
              titles=["swarm Plot", "Srip Plot", "Boxen Plot", "Box Plot"],
             major_title='Swarm and Boxen plots Vs. Strip and Box plots',
             style=None)

plt.show()


# This case is one of the scenarios where it is better to use swarmplot than to use striplot, because the tips dataset is rather small. 

# #### (II) Two numerical features & one categorical feature
# With the Seaborn function, **jointplot**, we are able to to plot at least two variables with bivariate and univariate graphs at once. \
# This function creates a Seaborn JointGrid object with multiple plot kinds (which makes jointplot to be a lightweight wrapper of JointGrid).
# 
# I used jointplot in order to plot 2 numerical features and 1 categorical feature.

# In[65]:


def reg_dist_plot(df, x_col: str, 
                  y_col: str, hue: str,
                  major_title = None,
                  figsize= (8,15),
                  palette= "Set2"): 
    sns.set_palette(palette)
    g = sns.jointplot(df[x_col], df[y_col], kind='reg', height=8, ratio=3, 
                      marginal_kws=dict(edgecolor="w"), scatter=False)
    sns.scatterplot(df[x_col], df[y_col], hue=df[hue], palette=palette, s=40)
    if major_title:
        g.fig.suptitle(major_title, y=1.02, fontsize=17)
    g.ax_joint.set_xlabel(x_col, fontsize=12)
    g.ax_joint.set_ylabel(y_col, fontsize=12)
    plt.legend(title=hue, loc=0,  fontsize='large')


# In[66]:


reg_dist_plot(titanic, "fare", "age", "pclass", "Fare to Age ratio across Passenger Class")

plt.show()        


# Using jointplot, I plotted the 2 numerical features of the titanic dataset, "fare" and "age", and "pclass" , a categorical feature, as well. \
# I decided to display a regplot that will present the relationship between the two categorical features, \
# while the color of the point represents the categorical feature. \
# Additionally, a distplot was formed for each of the numerical features (same axis). \
# 
# From the figure above we can gather a few conclusions: 
# - First, without much of a surprise, passengers of the first class have more fare compared to the second class and \
# even more compared to passengers in the third class. 
# - Secondly, there is a higher percentage of young passengers in the third class compared with the first class. \
# Also, from the distplot of the "fare" feature, we see that the distribution have a long-right-tail (a clear right-skewed distribution).

# ### 4. Tree Maps
# Treemap is a good visual technique for displaying a large and hierarchical dataset. \
# This type of method fills all available space of a figure with a hierarchy of rectangles of different sizes, \
# while creating a sort of nested rectangles. \
# Up until now, it sounds like treemaping works similarly to simple tree diagrams (where each rectangle is a branch) - \
# we will not cover tree diagrams in this notebook. \
# What makes treemapping unique is the fact that the area of each of the rectangles is proportional to the size of data values that it represents. \
# In most cases, color is added as another differentiation factor that helps display the data with a higher dimensionality. \
# We can say that treemaps are some special combination between tree diagrams and pie charts. 
# 
# 
# Treemaps grant the reader the ability to understand the dataset hierarchy and to identify patterns in the data. \
# However, like pie charts, they wouldn't be useful in displaying minor comparison subtlety. \
# And it goes without saying, but the major disadvantage of this type of chart is that it is good for categorical features. \
# In case of ordinal feature values, by default, the order of the rectangle of each value will be presented in ascending order \
# based on each value name (and not the count).
# 
# 
# We will explore 2 additional famous modules that are efficient for creating treemaps: squarify and plotly.

# #### (I) squarify
# This library is dedicated to calculate a treemap layout based on Squarified treemapping algorithm.
# 
# 
# A use case where it if useful to plot a treemap with squarify is with the Airbnb dataset.

# In[67]:


# !pip install squarify


# In[68]:


import squarify


# In[69]:


data, col, limit, palette, figsize = listings, 'property_type', 7, "Paired", (15,3)

plt.figure(figsize=figsize)
colors = list(sns.color_palette(palette))[:data[col].nunique()] if not limit else list(sns.color_palette(palette))[:limit] 
checked = pd.DataFrame(data[col].value_counts()) if not limit else pd.DataFrame(data[col].value_counts()[:limit])
squarify.plot(sizes=checked[col], label=checked.index, 
              alpha=0.5, color=colors ,text_kwargs={'fontsize':10})
plt.axis('off');


# The treemap above is presenting the listing property types distribution. Based on the percentage of the type out of all the listings, \
# the size of the rectangle is determined. 
# 
# 
# From this example, we can gather that the most common listing type is apartments. The second most common is a house, and \
# third will probably be a boat. \
# Notice how the values of rectangles are sorted alphabetically and not based on their size.
# 
# 
# This is a very simple presentation of a treemap, which is not much different than a regular pie chart.

# #### (II) plotly
# Plotly offers to create dynamic and interactive charts, where the reader could be presented with additional information \
# just by moving the mouse cursor intuitively to the location of a certain value. \
# Beside treemaps, plotly offers a range of unique chart types, that is good for showcasing numerous types of features and datasets. \
# I highly recommend checking plotly documentation for additional chart types.
# 
# 
# As an example, we will use the tips dataset:

# In[70]:


import plotly.express as px # if you hadn't already imported plotly.express


# In[71]:


tips["0"] = "root" # we are creating an additional dummy feature in order to have a single root node
tips['smoker'] = tips['smoker'].map({'No': 'Not a Smoker', 
                                     'Yes':'Smoker'}) # values that will make more sense on the treemap

cat_col_tips = ['0'] + [x for x in tips.select_dtypes(include='object') if x != '0']
fig = px.treemap(tips, path=cat_col_tips, values='tip',
                  color='total_bill', 
                  color_continuous_scale='tempo',
                  color_continuous_midpoint=np.average(tips['total_bill'], 
                                                       weights=tips['tip']))
fig.show()


# _(**hint:** try to move the mouse cursor and click on diffrent values)_

# This treemap is more advanced than the previous treemap, because it displays higher complexity. \
# Now we can observe the nested-rectangles characteristic of treemaps as we discussed earlier. \
# The size of each rectangle is proportional to the value size in the parent rectangle.
# 
# 
# In this example we added the color element to showcase additional dimensionality by featuring a numeric variable as well ("total_bill"). \
# Out of this chart, we are granted with much more information than we did in the previous treemap example. \
# For e.g, on Sundays, most of the clients are male. Over dinner, out of those males, the majority does not smoke. \
# Interestingly, the smokers portion of the male clientele over Sunday's dinners are tend to leave larger tips (or at least to pay more).

# ### 5. Seaborn FacetGrid object
# Up until now we demonstrated how to create your own advanced plotting functions and how to present multidimensional data \
# in one figure simultaneously. In order to do so, we used numerous lines of code. \
# This is where **Seaborn's FacetGrid** object comes into place.
# 
# 
# A FacetGrid object allows us to create multiple axes in a figure (a grid of subplots), that stores the information of our data. \
# Each of the axes contain a different plot, so we are able to showcase multiple types of charts at once. \
# Each of those plots are also referred to as "lattice" (or "trellis" or "small-multiple") graphics. \
# Using FacetGrid we can transfer the structure of the dataset into subplots, so it will be better visualized.
# 
# 
# Generally, in order to create a FacetGrid object, while initializing it, we need to choose the dataset and the features \
# that will structure the object. Afterward, it is possible to pick multiple plotting functions and apply them on the object \
# by using some map function. \
# This all process is done by just a couple lines of code, making it very easy to use. \
# It is optional to create a FacetGrid object directly (like we did in the first 2 Joyplots examples), \
# or to use another function that creates this type of object. \
# In this article, we will elaborate on 2 functions that do just that.

# #### (I) CatPlot
# The catplot function acts as an interface into creating a FacetGrid object that will present data of one or more \
# categorical features and a numerical feature. \
# This is done while presenting the relationship between the features in several ways (different plots).
# 
# Using the "kind" parameter we are able to select the plot type 
# 
# _(**FYI:** stripplot is the default option in the catplot function)._

# In[72]:


sns.set_theme(style="white"), sns.set_palette("Set2")

g = sns.catplot(x="sex", y="total_bill",
                hue="smoker", col="day",
                data=tips, kind="violin",
                height=4, aspect=.7);


# The example above uses the tips dataset to display the relationship between 3 categorical features, "sex" "smoker" and "day", \
# and 1 numerical feature, "total_bill". \
# For each weekday, a violinplot was created. On the x-axis we have the the two values of the sex feature. \
# For each value of the "sex" feature there are 2 violins: one for no-smoker and one for smoker (the "smoker" feature values).
# 
# 
# This type of multi-plotted figure gives the reader a lot of insight about the data. \
# For instance, we can conclude that female smokers tend to pay less over Fridays, both compared to non-smoker females and \
# compared to every other day of the week.

# #### (II) PairPlot
# The Seaborn's pairplot function is good when we wish to plot a pairwise relationship in the dataset. 
# 
# 
# By calling this function, a FacetGrid object will be created. \
# Both the x-axis and y-axis will display a row and a column with each of the dataset numeric features. This way, \
# the distribution of each numeric feature across every other numeric feature will be presented. \
# This type grid structure leaves the diagonal plots with the same vaules for the x and y axis (same feature) - \
# hence they are treated differently (univariate plot types will be applied instead). \
# This type of plotting technique can be extremely beneficial at the EDA phase of the data for example.
# 
# 
# Another option is to add additional categorical feature to the mix under the "hue" parameter, like I did in the following example:

# In[73]:


sns.set_theme(style="white"), sns.set_palette("Set2")

g = sns.pairplot(tips, hue="smoker", corner=True);


# All 3 of the numerical features in the tips dataset were automatically applied into a grid of subplots. \
# The categorical feature, "smoker", was also applied by the use of different colors.
# 
# 
# Those types of figures are very informative. We can explore the data very fast and efficiently. It should be emphasized \
# that we got those results by just 2 lines of code, but we are already able to notice very specific insight from the data. \
# For instance: When looking at the subplot of tip to size, we can see that usually people come in small party sizes and leave small tips. \
# But people that scones in bigger sized parties usually leave more tip. We can also see that usually, people that come with \
# large parties will usually be smokers.
# 
# 
# The diagonal plots showcase the distribution of one of the numerical features by the categorical feature. \
# As we can see, the type of the diagonal plots is a KDE plot.

# Now let's use a dataset with more features, like the Boston Housing dataset:

# In[74]:


# sns.set_theme(style="white")

g = sns.PairGrid(boston.iloc[:, 5:10], palette=["y"])
g.map_upper(plt.scatter, s=10, color="steelblue")
g.map_diag(sns.histplot, color="cadetblue")
g.map_lower(sns.kdeplot, cmap="ch:2.5,-.2,dark=.3") 

plt.show()


# In this example we are looking only at the relationship between numerical features. 
# 
# 
# Here, for every couple of features there are 2 types of plots: a bivariate KDE plot (over the lower plots) and \
# a scatter plot (across the upper plots). The "middle" plots, the diagonal plots, were presented as histograms \(because it's a univariate plot).
# 
# 
# From this graph, we can extract many pieces of information. \
# Let's look over the plots of "TAX" and "RAD". The former column presents the property tax rate and the latter refers to \
# the index of accessibility to radial highways. \
# The scatter plot helps us to see that in cases of high tax rate, the RAD index is especially high - meaning better accessibility to highways. \
# The scatter plot gives us an additional insight - we can see that cases of extremely high tax rates are probably outliers. \
# But if we look at the scatter plot of "TAX" to "DIS", the weighted mean of distances to five Boston employment centers, \
# we can see many more samples with the "TAX" extreme values. \
# Samples with tax-rate extreme values will usually have relatively low "DIS" value - meaning they will be closer to the employment centers. \
# Usually employment centers are found in central areas so that they will be as accessible as possible to the public. \
# On the other hand, from the bivariate KDE plot, we can gather that in the data there are at least 2 sub-populations: \
# 2 clusters of samples that act differently. So the samples with the extreme-tax rates are a subgroup on their own.

# In[75]:


# sns.set_theme(style="white"), sns.set_palette("Set2")

g = sns.pairplot(boston.iloc[:, 5:10], diag_kind="kde", corner=True)
g.map_lower(sns.kdeplot, levels=4, color=".2")

plt.show()


# The above figure displays the exact same information as the figure before. The only difference is that here the number of subplots \
# had been reduced so the figure seems less crowded. \
# Presentation of information in such a way, makes it look easier on the eyes and more readable.
# 
# 
# What we have here is basically the same scatterplots and density plots as before presented one on top of the other. \
# We were able to do so just by not mapping the upper plots of the grid - meaning this presentation structure is the default \
# setting for pairplot function.
# 
# 
# This time the diagonal plots were presented in a KDE plot as well. \
# If we'll go back to our "TAX" example, we can see yet again that just by the plot of the "TAX" feature alone, \
# there is a bimodal distribution. This type of distribution could have been our first clue that would have suggested the \
# possibility of 2 subgroups in the given populations.

# ## Summary:
# In this notebook we covered additional ways to visualize and present different types of data. 
# As was stated in the previous article, [**Data Visualization 101 - Part I**](https://nofar-herman.medium.com/data-visualization-101-part-i-8c0b1b473a49), data visualization is a tool we can use to tell a story. \
# Your job as the storytellers is to choose the plot which will be best suited for this job.
# 
# 
# I hope that both the notebooks and the articles provided you with a good start to data visualization. \
# There are many more methods to explore data which we did not cover today. I believe that the best way to learn something new \
# is to just go at it. So I encourage you to try those functions for yourselves!
# 
# 
# If you would like to see other data visualization techniques as well, you're welcome to ⭐ [this repo](https://github.com/nofr) and check the code from all my articles \
# **+** [Follow me on Medium](https://nofar-herman.medium.com/) to stay informed with my new data visualization and data science articles.

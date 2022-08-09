#!/usr/bin/env python
# coding: utf-8

# # data visualization 101 - part I : The Notebook
# Basic visualization functions

# ### import packages:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# ### creating and dataframes:

# #### (I) Iris dataset:

# In[2]:


from sklearn.datasets import load_iris

iris_data = load_iris()
print(iris_data.DESCR)

X_iris, y_iris = load_iris(return_X_y=True, as_frame=True)
iris = X_iris.join(y_iris.rename("species"))
display(iris.sample(5))


iris_target_names = {i:name for i, name in enumerate(iris_data.target_names)}
for k, v in iris_target_names.items():
    print(f"At the target feature 'species', {k} means {v}")


# #### (II) Boston Housing dataset:

# In[3]:


from sklearn.datasets import load_boston

boston_data = load_boston()
print(boston_data.DESCR)

x_bos = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
y_bos = pd.Series(boston_data.target, name="PRICE")
boston = x_bos.join(y_bos)
display(boston.sample(5))


# #### (III) Tips dataset:

# In[4]:


import plotly.express as px

tips = px.data.tips()
target = "tip"
features = [col for col in tips.columns if col != target]
x_tips, y_tips = tips[features], tips[target]
tips = x_tips.join(y_tips)
tips.sample(5)


# #### (IV) Titanic dataset:
# (this time I used an excel sheet)

# In[5]:


titanic = pd.read_excel('titanic3.xls').drop_duplicates()
titanic.sample(5)


# ## Ploting:

# ### 1.) choosing the right graph
# **basic countplot, barplot and histogram**

# When we first come across a dataset, we should be able to classify each feature type, \
# categorical (nominal and ordinal) or numerical (discrete and continuous). \
# For each type of feature there are different types of graph which can be helpful. \
# For example, with the iris dataset, the target feature is "species", which is a nominal categorical feature. \
# Another feature of the iris dataset is "petal length (cm)", a continuous numerical feature.
# \
# In order to examine each of those features' distributions, we should use different graphs. \
# But if we would need to see how one feature affects the other one, we will need to use a different kind of graph altogether.

# In[6]:


def count_bar_hist(df, cat_col: str, num_col: str,
                   cat_col_labels: list = None,
                   titles: list = None,
                   major_title: str = None,
                   palette: str = "Set3", 
                   figsize: tuple = (12,4)):

    color = list(sns.color_palette(palette))[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    ax1 = sns.countplot(x=cat_col, data=df, palette=palette, ax=ax1)
    ax2 = sns.barplot(x=cat_col, y=num_col, data=df, palette=palette, ax=ax2)
    ax3 = sns.histplot(x=num_col, data=df, color=color, edgecolor="w", lw=2, ax=ax3)
    if cat_col_labels:
        ax1.set_xticklabels(cat_col_labels)
        ax2.set_xticklabels(cat_col_labels)
    if titles:
        ax1.set_title(titles[0], fontsize=13)
        ax2.set_title(titles[1], fontsize=13)
        ax3.set_title(titles[2], fontsize=13)
    if major_title:
        plt.suptitle(major_title, y=1.1, fontsize=15, fontweight='bold')
    plt.tight_layout();


# * For the distribution of a numerical feature it is common to use Histogram for example. Histogram is a frequency distribution plot where the x-axis contains the total range of the variable values, and each bin ("bar") contains a specified sub-range of values. The y-axis represents the frequency number for each bin. Histograms are highly used because they are very easy to understand. \
# \
#  \
#  
# * For the distribution of a categorical feature, it is recommended to use a Count Plot. A count plot is a method that helps to show the total counts of observations in each categorical bin using bars. In some way, it is a histogram for categorical features, where the x-axis contains all the values of the categorical feature and the y-axis shows the number of times that this value appeared. \
# \
#  \
#  
# * When we wish to evaluate how 1 feature might alter another feature, we could use a Bar Plot. That is true for the most part, when at least one feature is categorical.

# In[7]:


count_bar_hist(iris, "species", "petal length (cm)", iris_target_names.values(), 
              ["Count Plot\nof the target feature",
               "Bar Plot\nof the target to another feature",
              "Histogram\nof a numeric feature"],
              major_title = "3 Types of basic distribution plots")


# This function simply helps to organize the 3 plot types neatly in a row, but we can simply use just part of that function code in order to plot just one kind of those graphs.

# ### 2.) Distribution of 1 numeric feature
# **Histogram and kde plot**

# To showcase the distribution of a numerical feature we can use, as mentioned earlier, \
# a Histogram, or even a KDE (Kernel Density Estimate) plot.
# Similar to a histogram, the values range in the x-axis of a KDE plot is of the variable values range, \
# but the y-axis represents the probability density, this way a KDE plot presents a probability density curve of the data. \
# One of the advantages of KDE plot is that unlike Histogram, we don't lose information due to binning. \
# Histograms can also have density at their y-axis (simply change stat to "density" at seaborn.histplot as shown at the code below).

# In[8]:


def hist_kde_dist(df, col: str,
                  titles: list = None,
                  major_title: str = None,
                  colors: list = [],
                  palette: str = "Set3",
                  diff_colors = False,
                  figsize: tuple = (12,4)):
    n = 1
    if diff_colors or len(colors)>1:
        n += 1 
    if not colors:
        colors = list(sns.color_palette(palette))[::2][:n] # to avoid the light yellow color. not a must
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=figsize)
    
    ax1 = sns.histplot(x=col, data=df, color=colors[0], edgecolor="w", bins=5,
                       lw=2, stat="density", ax=ax1)
    ax2 = sns.kdeplot(x=col, data=df, color=colors[-1], lw=2, ax=ax2)
    ax3 = sns.distplot(df[col], hist=True, color=colors[0], ax=ax3,
                       hist_kws=dict(edgecolor="w", linewidth=2), 
                       kde_kws=dict(linewidth=2, color=colors[-1]))
    if titles:
        ax1.set_title(titles[0], fontsize=13)
        ax2.set_title(titles[1], fontsize=13)
        ax3.set_title(titles[2], fontsize=13)
    if major_title:
        plt.suptitle(major_title, y=1.1, fontsize=15, fontweight='bold')
    plt.tight_layout();


# A seaborn.distplot is a plot that presents a histogram (with density at the y-value) and a KDE curve on top of it. \
# That way you could get the benefits of both Histogram and a KDE plot.

# In[9]:


hist_kde_dist(iris, "petal length (cm)", 
              ["Histogram (Density)", "KDE plot", "Distplot"],
              major_title = "3 Basic Types of Numeric Feature Distribution Plots")


# Another option is to male the KDE line in a different color:

# In[10]:


hist_kde_dist(iris, "petal length (cm)", 
              ["Histogram (Density)", "KDE plot", "Distplot"],
              major_title = "3 Basic Types of Numeric Feature Distribution Plots",
             diff_colors=True)


# _At the histogram (seaborn), I chose chose `stat="density"` over the defult "count" to see how distplot combains the histogram with the KDE plot_

# ### 3.) Distribution of 1 categorical feature
# **ring pie chart**

# When it comes to categorical data distribution, beside count plot, a good old Pie Chart can do the trick. \
# On top of the pie chart it's very common to add annotations at the percentages of each value. \
# Additionally, we can upgrade the classic appearance of the pie chart in different ways, \
# such as making it in a ring shape (aka donut chart), like at the following example:

# In[11]:


def ring_pie_chart(df, col: str,
                    col_labels: dict = None,
                    title: str = None,
                    explode: bool = None,
                    palette: str = "Set3",
                    ax = None,
                    figsize: tuple = (4,4)):
    # preparing the values for the pie chart:
    values, counts = zip(*dict(df[col].value_counts().sort_index()).items())
    if col_labels:
        if len(col_labels) == len(values):
            values = [col_labels[val] for val in values]
    colors = list(sns.color_palette(palette))[:len(values)]
    # creating the explode (the ring gaps):
    if explode:
        explode = tuple([0.05] * len(values))
    elif explode is False:
        explode=None
    # create the pie chart itself:
    if not ax:
        f, ax = plt.subplots()
    ax.pie(counts, colors=colors, labels=values, autopct='%1.1f%%', startangle=90, explode=explode)
    # creating the ring (the inner circle):
    inner_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(inner_circle)
    ax.axis('equal') 
    if title:
        ax.set_title(title, y=1.08, fontsize=15)
    plt.tight_layout()


# In[12]:


ring_pie_chart(tips, "day",
                {'Sun': "Sunday", 'Sat': "Saturday", 
                 'Thur': "Thursday", 'Fri': "Friday"},
                title = "Round Pie Chart")

plt.show()


# We can even create "gaps" between the different parts of the pie chart while adding the "explode" parameter to the matplotlib.pie function.

# In[13]:


ring_pie_chart(tips, "day",
                title = "Ring-shape Pie Chart With Gapping",
                explode=True)

plt.show()


# New research has shown that people tend to react better to data that is presented in a count plot (bar plots in general) compared to pie charts. \
# The reason for that is that people struggle to understand which of the parts is greater and by how much, unlike bar plots, \
# which are very easy to distinguish each group size and by how much they differ. \
# We can conclude that a bar chart can be beneficial in precise evaluation of the size and percentage of each part. \
# Alongside a pie chart is usually used if the sum of all same-value-cases add up to a meaningful size, therefore it is mainly built to visualize the contributions of each part on the whole.

# **different ways to plot 1 categorical data**

# In[17]:


palette, style = "Set3", "white"
df, cat_col = tips, "day"

sns.set_theme(style=style)
figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
figure.suptitle("Different ways to visualize categorical data", y=1.08, fontsize=20)

ax1 = sns.countplot(x=cat_col, data=df, palette=palette, ax=ax1)
ax1.set_title("classic Bar Plot", y=1.08, fontsize=15)
# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

values, counts = zip(*dict(tips['day'].value_counts().sort_index()).items())
colors = list(sns.color_palette(palette))[:len(values)]
ax2.pie(counts, colors=colors, labels=values)
ax2.set_title("classic Pie Chart", y =1.08, fontsize=15)

ring_pie_chart(df, cat_col, ax=ax3, title="ring shaped Pie Chart")

plt.tight_layout();


# At the end, to pick which of the two (or three) is better is your choice to make.

# ### 4.) Plots for only 2 categorical features

# Sometimes we will wish to demonstrate the distribution of 2 categorical features at the same time. \
# In this case we will have a few options:
# * The Side-by-Side Bar Chart (also known as grouped or double bar chart) is used to show how the data is distributed across different values of 2 categorical features, unlike the classic bar chart where there is only 1 categorical feature. This kind of graph allows exhibition of a primary and a secondary distribution of data, so we can see how the second categorical feature changes within each value of the first categorical feature. \
# \
# \
# 
# * Stacked Bar Chart is very similar to the double bar chart, only instead of a secondary level bar chart of one feature for each value of the second feature, each of the second feature bars is broken into colored-subsections that represent the proportion of the first feature values. In a certain way, it is kind of a combination between a bar chart on the primary level and a pie chart on the secondary level. So on the one hand, this kind of bar is easier-on-the-eyes because it has less bars on the horizontal axis. But on the other hand, like in a pie chart, the second level distribution becomes more difficult to understand and compare. \
# \
# \
# 
# * The Nested Pie Chart (aka double or multi-level pie chart) is an advanced representation of a classic pie chart. This kind of chart contains a set of concentric rings, where the sizes of the secondary features values are proportional to the total size of each value of the primary features. Similar to a nested pie chart, there is a Nested Donut Chart (also called multi-level doughnut chart), which in this article we will still refer to as a double pie chart (because this is what it is basically). Additionally, this kind of chart doesn't have to display simply 2 features as presented here, but as the number of layers is higher so is it harder for the reader to understand the chart. Therefore I will recommend to use only up to 2 features comparison at once.

# **Nested pie chart**

# In[14]:


def double_pie_chart(df, main_col: str, sub_col: str,
                     main_col_labels: dict = None,
                     title: str = None,
                     ax = None,
                     palette1: str = "Set2", 
                     palette2: str = "Pastel1", 
                     figsize: tuple = (10,5)):
    # creating the chart labels and weights:
    values, counts = zip(*dict(df[main_col].value_counts().sort_index()).items())
    if main_col_labels:
        if main_col_labels == len(values):
            values = [main_col_labels[val] for val in values]
    sub_values = list(set(df[sub_col]))
    sub_values.sort()
    sub_values_dict = {}
    for val in values:
        sub_values_dict[val] = dict(df.loc[df[main_col] == val, sub_col].value_counts().sort_index())
        if len(sub_values_dict[val]) < len(sub_values):
            for l, sv in enumerate(sub_values):
                sub_values_dict[val][sub_values[l]] = sub_values_dict[val].get(sub_values[l], 0)
        sub_values_dict[val] = list(sub_values_dict[val].values())
    sub_counts = [c for pair in sub_values_dict.values() for c in pair]
    sub_values = sub_values * len(values)
    colors = list(sns.color_palette(palette1))[:len(values)]
    sub_colors = list(sns.color_palette(palette2))[:len(set(sub_values))]

    # explode
    explode = tuple([0.05] * len(values))
    sub_explode = tuple([0.05] * len(sub_values))

    # plotting
    if not ax:
        f, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(counts, colors=colors, startangle=90, frame=True, explode=explode,
                           wedgeprops=dict(width=0.9))
    ax.pie(sub_counts, colors=sub_colors, radius=0.65, startangle=90, explode=sub_explode,
           autopct='%1.1f%%', textprops={'fontsize': int(figsize[0]*0.8), 'rotation': 40}, pctdistance=0.85)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(values[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)

    inner_circle = plt.Circle((0,0),0.5,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(inner_circle)
    plt.axis('equal')
    if title:
        ax.set_title(title, y=1.1, fontsize=figsize[0]*1.5)
    plt.tight_layout()


# In[15]:


double_pie_chart(tips, "day", "sex", title= "A Double (Nested) Pie Chart");


# **advanced plots for categorical data (multi catagorical features)**

# In[16]:


figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

figure.suptitle("Diffrent ways to visualize multi-featured categorical data", y=1.08, fontsize=20)
ax1 = sns.countplot(x="day", hue="sex", data=tips, palette="Pastel1", ax=ax1)
ax1.set_title("Side-by-side Bar plot", y=1.1, fontsize=15)

# Hide the right and top spines
ax1.spines['right'].set_visible(False) 
ax1.spines['top'].set_visible(False) 

ax2 = sns.histplot(data=tips, x="day", hue="sex", multiple="stack", palette="Pastel1", 
                   edgecolor="w", lw=20, ax=ax2)
ax2.set_title("Stacked Bar plot (Barh)", y=1.1, fontsize=15)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.get_legend().remove()
double_pie_chart(tips, "day", "sex", ax=ax3, title="Double Pie chart")
plt.tight_layout();


# Knowing all types of plots strengths and weaknesses will help us determine which of the plots will be better suited to our needs.

# ### 5.) bar plots comparison for different DataFrames

# In many cases we would like to split our data to train and test sets or train validation and test sets. \
# The split can be done in many ways, such as: time series split, random split, stratified split, and so on. \
# Due to the split technique we choose, the data may not be evenly distributed at all the features - which may affect our model performance. \
# For that reason, we should check for even distribution across all data sets. \
# Let's take the following example: we would create 2 train-dev-test sets from the titanic dataset, each split by a different method  - \
# one by a random split and the other one by stratified of the sex feature split.

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


target = 'survived'
features = [c for c in titanic.columns if c != target]
X, y = titanic[features], titanic[target]


# In[20]:


X_train_ran, X_test_r, y_train_ran, y_test_r = train_test_split(X, y, train_size=0.7, random_state=42)

X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_train_ran, y_train_ran, train_size=0.7, random_state=42)


# In[21]:


X_train_stra, X_test_s, y_train_stra, y_test_s = train_test_split(X, y, train_size=0.7,
                                                                  random_state=42, stratify=titanic['sex'])

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X_train_stra, y_train_stra, 
                                                          train_size=0.7, random_state=42, stratify=y_train_stra)


# In[22]:


# randomly splited titanic datasets:
rand_set = [[X_train_r, X_val_r, X_test_r],
            [y_train_r, y_val_r, y_test_r]]

# stratify splitted by sex titanic datasets:
strat_set = [[X_train_s, X_val_s, X_test_s],
             [y_train_s, y_val_s, y_test_s]]


# **simple comparison bar chart**
# 
# To simply see if the data was evenly distributed between all data sets of both splits, we can use the following function that will present the results as a bar plot.

# In[23]:


def plot_bar_compare(datasets,
                     col:str,
                     dataset_names=None,
                     title=None,
                     rot=False,
                     colors=None,
                     figsize=(9,3)):
    """ Compare the distribution between train, validate and test datasets """
    fig, axs = plt.subplots(1, len(datasets), figsize=figsize, sharey=True)
    if not colors:
        colors = ['yellowgreen', 'sandybrown', 'rosybrown', 'tomato']
    if title:
        fig.suptitle(title, y=1.12, fontsize=17)
    for i, set in enumerate(datasets):
        if len(datasets) == 2:
            if not dataset_names:
                dataset_names = ["train", "test"]
        elif len(datasets) == 3:
            if not dataset_names:
                dataset_names = ["train", "validation", "test"]
        labels= datasets[i][col].value_counts().sort_index()
        dict_names = dict(zip(labels.keys(),((100 * (labels) / len(datasets[i].index)).tolist())))
        names = list(dict_names.keys())
        values = list(dict_names.values())

        axs[i].bar(names, values, color=colors[i])
        axs[i].grid()
        axs[i].set_title(f'{dataset_names[i].title()} data' if len(datasets) <= 3 else f'Data #{i+1}')
        if rot:
            axs[i].set_xticklabels(names, rotation=45)
    axs[0].set_ylabel('precentage (%)')
    plt.tight_layout()
    plt.show()


# For the purpose of that article, I choose to present the results for the "Embarked" column.

# In[25]:


plot_bar_compare(rand_set[0], 'embarked', title='Embarked distribution by random splitting', rot=True)

plot_bar_compare(strat_set[0], 'embarked', title='Embarked distribution by sex-based splitting', rot=True)


# So in our example, we can see that all data sets from the random split were divided evenly when looking at the "Embarked" feature. \
# This is not the case for the sex-based split data sets, that were evenly divided at the "Sex" feature, but not so much at the "Embarked" feature. \
# This kind of analysis can help us a lot in understanding how to choose the right model we should be working with, or better yet, \
# to make an educated decision with how to split or handle our data.

# **grouped comparison bar chart** (at the same dataframe)
# 
# In the last example we compared the train validation and test sets of both splits. \
# In the next example, we will check the data distribution between the datasets of the same split, and then we will set the two against each other. \
# Hence we will use a grouped bar plot. \
# Just to make it easier for our analysis, we can merge all the same-split data sets into 1 whole DataFrame (like it used to be originally), \
# only this time we added a column ("group") that claims each sample to it's previous dataset, like this:

# In[27]:


X_train_r["group"], X_val_r["group"], X_test_r["group"] = 'Train', 'Validation', 'Test'
X_train_s["group"], X_val_s["group"], X_test_s["group"] = 'Train', 'Validation', 'Test'

random_split_df = pd.concat([X_train_r, X_val_r, X_test_r])
strat_split_df = pd.concat([X_train_s, X_val_s, X_test_s])

print("Random:")
display(random_split_df.sample(2))
print("Stratify:")
display(strat_split_df.sample(2))


# In[32]:


def bar_comparison_same_data(datasets:list,col:str, hue:str,
                             dataset_names=None,
                             major_title=None,
                             palette=None,
                             colors=None,
                             figsize=None):
    if not figsize:
        figsize = (5*(len(datasets)), 5)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    if not colors and not palette:
        palette = "Set3"
    elif palette:
        colors=None
    elif colors and not palette:
        palette = colors
    if major_title:
        fig.suptitle(major_title, y=1.12, fontsize=17)
    for i, ax in enumerate(axs.flat):
        ax = sns.countplot(x=col, hue=hue, data=datasets[i], ax=axs[i], palette=palette,
                           # in order to keep same x-axis values order:
                           order=sorted(datasets[i][col].dropna().unique().tolist())) 
        ax.grid()
        ax.set_title(dataset_names[i] if dataset_names else f"data #{i+1}")
    plt.tight_layout()


# In[33]:


bar_comparison_same_data([strat_split_df, random_split_df], 'embarked', 'group', ["Sex-based split", 'Random split'],
                         major_title="Different split types comparison");


# Here we can see a slightly better that the 2 splits gave similar results for the "Embarked" feature, but there are a few variations between the the datasets, where the variations in the sex-based split datasets are more notable.

# **EXTRA: distplot**

# In[30]:


def distplots(df, col: str, target: str,
              major_title= None,
              figsize= None,
              color= None):
    values = df[col].dropna().unique()
    if not figsize:
        figsize = (5, 3 * len(values))
    fig, axs = plt.subplots(len(values), 1, figsize=figsize)
    for i, ax in enumerate(axs.flat):
        sns.distplot(df.loc[df[col] != values[i], target], ax=axs[i],
                     color="cornflowerblue" if not color else color)
        axs[i].set_title(f"{col} : {values[i]}")
    if major_title:
        plt.suptitle(major_title, y=1.02, fontsize=15)
    plt.tight_layout()
    plt.show()


# In[31]:


distplots(strat_split_df, col="embarked", target="fare", 
          major_title="Distribution of embarked values at the sex-based split set")


# ### 6.) Plots for 2 numeric features

# When we have a pair of numerical features, and we want to see if they are related or to examine how one numerical feature \
# influences the other, there are numerous plotting options. \
# That being the case, we can use one of the following plot: \
# \
# 
# * A Scatter Plot is a common way to display the relationship between different numerical features. This kind of plot can be extremely helpful when we want to determine whether or not there is any kind of correlation or some kind of a pattern between the two numerical features. The data is presented by x and y coordinates for each sample. The x-axis value for each point will be determined by the value of the first numerical feature of that sample, and the y-axis value will be determined by the second numerical feature accordingly. Ergo the name "Scatter" - because all data points seem like they are scattered across the graph. \
# \
# \
# 
# * A Regression Line shows the overall trend of the data. It is based on a statistical method that assists with modeling the relation between 2 numerical features. This plot is formed on the basics of a scatter plot data (on several x,y data points). In general the regression line helps to estimate points on the graph where part of the data is missing. The main advantage of this plot is that it smoothen the noise that we get in a scatter plot so we can see a clear linear trend of the data. \
# \
# \
# 
# * Regression lines are many times added on top of another plot (as some kind of annotation). The most familiar format is the combination between a scatter plot and a regression line, which Seaborn called RegPlot. Thereby we can get both the general trend of the data without losing any information at the same time. \
# \
# \
# 
# Let's see an example for all three using the tips dataset:

# **scatter, line (regression), RegPlot**

# In[34]:


def scatter_line_regplot(df, num_col1: str,
                         num_col2: str,
                         titles: list = None,
                         major_title: str = None,
                         colors: list = [],
                         palette: str = "Set3",
                         diff_colors = False,
                         figsize: tuple = (12,4)):
    n = 1
    if diff_colors or len(colors)>1:
        n += 1 
    if not colors:
        colors = list(sns.color_palette(palette))[::2][:n] # to avoid the light yellow color. not a must
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=figsize)
    
    axs[0] = sns.scatterplot(data=df, x=num_col1, y=num_col2, color=colors[0], ax=axs[0])
    p = sns.regplot(data=tips, x="total_bill", y="tip", ax=axs[2],
                   scatter_kws={"color": colors[0], 'edgecolors': 'w'}, 
                    line_kws={"color": colors[-1]})
    x, y = p.get_lines()[0].get_xdata(), p.get_lines()[0].get_ydata()
    axs[1] = sns.lineplot(x, y, c=colors[-1], ax=axs[1])
    axs[1].set_xlabel('total_bill')
    axs[1].set_ylabel("tip")
    if titles:
        for i, ax in enumerate(axs):
            ax.set_title(titles[i], fontsize=13)
    if major_title:
        fig.suptitle(major_title, y=1.1, fontsize=15, fontweight='bold')
    plt.tight_layout()


# Everyting at the same color:

# In[35]:


scatter_line_regplot(tips, "total_bill", "tip",
                    ["Simple Scatter plot",
                    "Simple Regression line",
                    "Regplot: Scatter + Regression line"], 
                     "3 Basic Types of 2 Numeric Features Plots");


# Or different color for the scatter plot and the regression line plot:

# In[36]:


scatter_line_regplot(tips, "total_bill", "tip",
                    ["Simple Scatter plot",
                    "Simple Regression line",
                    "Regplot: Scatter + Regression line"], 
                     "3 Basic Types of 2 Numeric Features Plots", 
                     diff_colors=True)

plt.show()


# ### 7.) Multi numeric features scatter plot
# 

# The following function is an advanced scatter plot, where we can see the relationship between several numerical features. \
# As before, the x and y axis represents 2 numerical features, but now, the color and the size of the dot, \
# represents an additional 2 numeric features, which will result in a more complex and a fuller picture of how the data acts.

# In[42]:


def multi_feature_scatter(df, num_col1: str, num_col2: str,
                          num_col3: str = None,
                          num_col4: str = None,
                          cat_col: str = None,
                          title: str = None,
                          ax = None,
                          palette: str = "flare",
                          figsize: tuple = (7, 4)):
    """
    if num_col4 not given, a new feature will be assign - the rate (rati) between the 2 
    first numeric columns
    """
    if not ax:
        f, ax = plt.subplots(figsize=figsize)
    if not num_col4:
        rate = pd.Series(df[num_col2] / df[num_col1], name=f"{num_col2} to\n{num_col1} rate")
    sns.scatterplot(data=df, x=num_col1, y=num_col2,
                    hue=num_col4 if num_col4 else rate,
                    size=num_col3, style=cat_col, sizes=(20, 300) , # sizes=(20, (figsize[0]*figsize[1])*10),
                    palette=palette, legend="brief", ax=ax)
    ax.legend(loc=0,  bbox_to_anchor=(1, (figsize[0]+figsize[1])/10), labelspacing=1, prop={'size': 8})
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=15)


# In[43]:


multi_feature_scatter(iris, "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", 
                      title = "4 numeric features scatter plot")

plt.show()


# If we wish to include an additional categorical feature to this graph, we could do it as follows:

# In[44]:


multi_feature_scatter(tips, "total_bill", "tip", "size", cat_col= "sex", 
                      title = "5 features scatter plot:\n4 numerical features and 1 categorical")
plt.show()


# Now the categorical feature values are represented as different marks (O's and X's). \
# Now the scatter plot doesn't contain only numerical features, but a mixture of both numerical and categorical features. \
# \
# \
# 
# If you dudn't noticed the changes, maybe the following figure will help:

# In[45]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
multi_feature_scatter(iris, "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", 
                      title = "4 numeric features scatter plot", ax=ax1)
multi_feature_scatter(tips, "total_bill", "tip", "size", cat_col= "sex", 
                      title = "5 features scatter plot:\n4 numerical features and 1 categorical", ax=ax2)
plt.tight_layout();


# ### 9.) A cross-numeric features plot
# **Heatmap (corralation between numeric features)**

# Heatmaps are an easy-to-read visual tool that helps determine correlation between different features at the same time. \
# The strength and direction of each feature pair is portrayed by color from a colorbar. \
# This 2D correlation matrix is an important tool for data analysis, and it is useful especially while looking at multiple numeric features at once. \
# Checking the correlation between different features to the target feature or to simply check the correlation between all features \
# can help us with feature importance tasks and to give us a sense of what are the most relevant features to our case.

# In[46]:


def heatmap_corr(df, 
                columns: list = None,
                correlation: str = "spearman",
                title: str = None,
                style: str = "whitegrid",
                cmap = sns.diverging_palette(10, 100, 50, 50, as_cmap=True),
                font_scale: float = 1.5,
                x_ticks_rotation: int = 0, 
                y_ticks_rotation: int = 90, 
                figsize: tuple = None):
    sns.set_theme(style=style)
    if not columns:
        columns = df.describe().columns
    if not figsize:
        figsize = tuple([len(columns) + round(len(columns)/2)] *2)
    sns.set_style(style)
    plt.figure(figsize=figsize)
    corr = df[columns].corr() # the default is spearman
    if correlation:
        corr = df[columns].corr(method=correlation) # the default is spearman and not pearson

    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    ax = sns.heatmap(df.corr(), annot=True, fmt=".2f",mask=mask, square=True, cmap=cmap,
                     cbar_kws={"shrink": .7}, linewidth=5)
    plt.yticks(plt.yticks()[0], rotation=y_ticks_rotation)
    plt.xticks(plt.xticks()[0], rotation=x_ticks_rotation)
    if title:
        plt.title(title, loc='left', fontsize=15, fontweight="bold")
    sns.set(font_scale=font_scale)


# Let's use our Boston housing dataset to see an example for a feature correlation heat map:

# In[47]:


heatmap_corr(boston.iloc[:, :-1], y_ticks_rotation=0,
             title=' spearman correlation\n matrix'.upper())

plt.show()


# Now lets try it with a smaller portion of the Boston housing dataset:

# In[50]:


heatmap_corr(boston.iloc[:, :5], cmap='coolwarm',
             title=' spearman correlation\n matrix'.upper())

plt.show()


# Or even with a different correlation entirely:

# In[54]:


heatmap_corr(boston.iloc[:, :5], correlation="pearson", y_ticks_rotation=0,
             title=' pearson correlation\n matrix'.upper())

plt.show()


# _**NOTE:** good for numeric features ONLY._

# # DONE!
# 
# For more plot types and code for different plotting functions check out the next notebookes

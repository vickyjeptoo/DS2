import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML

df = pd.read_csv('https://gist.githubusercontent.com/johnburnmurdoch/4199dbe55095c3e13de8d5b2e5e5307a/raw/fa018b25c24b7b5f47fd0568937ff6c04e384786/city_populations',
    usecols=['name', 'group', 'year', 'value'])
print(df)
#We are interested to see top 10 values are a given year. Using pandas transformations, we will get top 10 values.
current_year = 2018
dff = (df[df['year'].eq(current_year)]
       .sort_values(by='value', ascending=True)
       .head(10))
print(dff)

#Now, let’s plot a basic bar chart.
# We start by creating a figure and an axes. Then, we use ax.barh(x, y) to draw horizontal barchart.
fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(dff['name'], dff['value'])
plt.show()
#color labels
colors = dict(zip(
    ['India', 'Europe', 'Asia', 'Latin America',
     'Middle East', 'North America', 'Africa'],
    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381',
     '#aafbff', '#f7bb5f', '#eafb50']))
group_lk = df.set_index('name')['group'].to_dict()
#group_lk is mapping between name and group values.
fig, ax = plt.subplots(figsize=(15, 8))
dff = dff[::-1]   # flip values from top to bottom
# pass colors values to `color=`
ax.barh(dff['name'], dff['value'], color=[colors[group_lk[x]] for x in dff['name']])
# iterate over the values to plot labels and values (Tokyo, Asia, 38194.2)
for i, (value, name) in enumerate(zip(dff['value'], dff['name'])):
    ax.text(value, i,     name,            ha='right')  # Tokyo: name
    ax.text(value, i-.25, group_lk[name],  ha='right')  # Asia: group name
    ax.text(value, i,     value,           ha='left')   # 38194.2: value
# Add year right middle portion of canvas
ax.text(1, 0.4, current_year, transform=ax.transAxes, size=46, ha='right')
ax.set_title('Most Popular Cities in the world from 1500 to 2018')
plt.show()
#We need to style following items:

    # Text: Update font sizes, color, orientation
    #Axis: Move X-axis to top, add color & subtitle
    #Grid: Add lines behind bars
    #Format: comma separated values and axes tickers
    #Add title, credits, gutter space
    #Remove: box frame, y-axis labels

#We’ll add another dozen lines of code for this.
fig, ax = plt.subplots(figsize=(15, 8))
def draw_barchart(year):
    dff = df[df['year'].eq(year)].sort_values(by='value', ascending=True).tail(10)
    ax.clear()
    ax.barh(dff['name'], dff['value'], color=[colors[group_lk[x]] for x in dff['name']])
    dx = dff['value'].max() / 200
    for i, (value, name) in enumerate(zip(dff['value'], dff['name'])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value-dx, i-.25, group_lk[name], size=10, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    # ... polished styles
    ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.text(0, 1.06, 'Population (thousands)', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'The most populous cities in the world from 1500 to 2018',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by @pratapvardhan; credit @jburnmurdoch', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
plt.box(False)
#ANIMATE RACE
#To animate the race, we will use FuncAnimation from matplotlib.animation.
# FuncAnimatio creates an animation by repeatedly calling a function (that draws on canvas).
animator = animation.FuncAnimation(fig, draw_barchart, frames=range(1968, 2019))
HTML(animator.to_jshtml())
plt.show()
#Bonus: xkcd-style!
#Turning your matplotlib plots into xkcd styled ones is pretty easy.
# You can simply turn on xkcdsketch-style drawing mode with plt.xkcd.
with plt.xkcd():
    fig, ax = plt.subplots(figsize=(15, 8))
draw_barchart(2018)
plt.show()
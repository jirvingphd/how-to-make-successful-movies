###### PLOTLY VERSIONS
import plotly.express as px
import plotly.io as pio

def set_template(template='seaborn'):
    pio.templates.default=template
    

def plotly_explore_numeric(df, x,width=1000, height=500):
    fig = px.histogram(df,x=x,marginal='box',title=f'Distribution of {x}', 
                      width=width, height=height)
    return fig


def plotly_explore_categorical(df, x ,width=1000, height=500):
    fig = px.histogram(df,x=x,color=x,title=f'Distribution of {x}', 
                         width=width, height=height)
    return fig
    

def plotly_numeric_vs_target(df, x, y='SalePrice', trendline='ols',add_hoverdata=True,
                            width=800, height=600,):
    if add_hoverdata == True:
        hover_data = list(df.drop(columns=[x,y]).columns)
    else: 
        hover_data = None
        
    pfig = px.scatter(df, x=x, y=y,# template='plotly_dark', 
                     hover_data=hover_data,
                      trendline=trendline,
                      trendline_color_override='red',
                     title=f"{x} vs. {y}",
                      width=width, height=height)

    
    pfig.update_traces(marker=dict(size=3),
                      line=dict(dash='dash'))
    return pfig
    

# def plotly_categorical_vs_target(df, x, y='SalePrice', agg='mean',width=800,height=500,):
#     if agg=='mean':
#         plot_df = df.groupby(x,as_index=False)[y].mean().sort_values(y, ascending=False)
        
#     elif agg=='median':
#         plot_df = df.groupby(x,as_index=False)[y].median().sort_values(y,ascending=False)
        
#     else:
#         plot_df = df
        
#     fig = px.bar(plot_df, x=x,y=y, color=x, title=f'Compare {agg.title()} {y} by {x}',
#                  width=width, height=height)

                
#     return fig

def plotly_categorical_vs_target(df, x, y='SalePrice', histfunc='avg', width=800,height=500):
    fig = px.histogram(df, x=x,y=y, color=x, width=width, height=height,
                       histfunc=histfunc, title=f'Compare {histfunc.title()} {y} by {x}')
    fig.update_layout(showlegend=False)
    return fig


def plotly_correlation(df, cmap='magma', cols=None):
    if cols == None:
        cols = df.columns
    corr = df[cols].corr(numeric_only=True).round(2)
    fig = px.imshow(corr, text_auto=True,width=600, height=600,
                    color_continuous_scale=cmap,
                    color_continuous_midpoint=0, 
                   title='Correlation Heatmap')
    return fig


# if __name__ == "__main__":
#     set_template()
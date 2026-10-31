from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
df=pd.read_csv(Path(__file__).parent/'data/spacex_launch_dash.csv')
app=Dash(__name__)
app.layout=html.Div([
 html.H1('SpaceX Launch Records Dashboard'),
 html.P('Historical 56-launch IBM course snapshot. Class 1 and Class 0 follow course labels.'),
 dcc.Dropdown(id='site',options=[{'label':'All sites','value':'ALL'}]+[{'label':s,'value':s} for s in sorted(df['Launch Site'].unique())],value='ALL',clearable=False),
 dcc.Graph(id='pie'),html.Label('Payload mass range (kg)'),
 dcc.RangeSlider(id='payload',min=0,max=10000,step=100,value=[0,10000],marks={i:str(i) for i in range(0,10001,2000)}),dcc.Graph(id='scatter')
],style={'maxWidth':'1150px','margin':'auto','fontFamily':'Arial'})
@app.callback(Output('pie','figure'),Output('scatter','figure'),Input('site','value'),Input('payload','value'))
def update(site,payload):
 data=df if site=='ALL' else df[df['Launch Site']==site]
 if site=='ALL':
  counts=data[data['class']==1].groupby('Launch Site').size().reset_index(name='count')
  pie=px.pie(counts,names='Launch Site',values='count',title='Share of successful launches by site')
 else:
  counts=data.groupby('class').size().reset_index(name='count');counts['Outcome']=counts['class'].map({0:'Class 0',1:'Class 1'})
  pie=px.pie(counts,names='Outcome',values='count',title=site+' outcomes')
 filtered=data[data['Payload Mass (kg)'].between(*payload)]
 scatter=px.scatter(filtered,x='Payload Mass (kg)',y='class',color='Booster Version Category',hover_data=['Launch Site','Flight Number'])
 for fig in [pie,scatter]:fig.update_layout(template='plotly_white')
 return pie,scatter
if __name__=='__main__':app.run(debug=False)

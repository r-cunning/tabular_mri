import dash
from dash import dcc, html, Input, Output
import os

# Assuming `reports` is your DataFrame containing the report data
# and SHAP plots are saved as "<fold_id>_shap_waterfall.png" in each fold's directory

# Base directory for logs and SHAP plots
base_directory = "lightning_logs"

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import dash_table
# Assuming the rest of your imports and parser setup goes here

app = dash.Dash(__name__)

reports = 2

app.layout = html.Div([
    dcc.Dropdown(
        id='fold-selector',
        options=[{'label': f"Fold {fold}", 'value': fold} for fold in reports.keys()],
        value=list(reports.keys())[0]  # Default to the first fold
    ),
    html.Div(id='report-output'),  # Placeholder for displaying the selected report as a DataTable
    html.Img(id='shap-image')  # Placeholder for displaying SHAP plot image
])

@app.callback(
    [Output('report-output', 'children'),
     Output('shap-image', 'src')],
    [Input('fold-selector', 'value')]
)
def update_content(selected_fold):
    # Update report
    report_df = reports[selected_fold]
    report_table = dash_table.DataTable(
        data=report_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in report_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'minWidth': '80px', 'width': '80px', 'maxWidth': '180px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'border': '1px solid black'
        },
        style_data={
            'border': '1px solid grey'
        }
    )
    
    # Update SHAP plot
    # Assuming SHAP plots are statically served from a directory accessible to Dash
    # In a real app, you'd need to set up static file serving if plots are not within Dash's assets folder
    shap_plot_path = shap_explanations[selected_fold]
    
    return report_table, app.get_asset_url(shap_plot_path)

if __name__ == '__main__':
    app.run_server(debug=True)
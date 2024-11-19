import dash
from dash import html, dcc, Input, Output,State, clientside_callback,callback
import dash_bootstrap_components as dbc



app = dash.Dash(__name__,use_pages=True,external_stylesheets=[dbc.themes.YETI,dbc.icons.FONT_AWESOME])



color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="switch"),
        dbc.Switch(id="switch", value=True, className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="switch"),
    ]
)


sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page["name"], className="ms-2"),
            ],
            href=page["path"],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    class_name="bg-light",
    
    
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("Análise PCA odor",
                         style={'fontSize': 50, 'textAlign': 'center'}))
     ]),
    
    
    html.Hr(),
    
    dbc.Row([
        color_mode_switch,   
        dbc.Col([
            sidebar
        ], xs=4,sm=4,md=2,lg=2,xl=2,xxl=2),
    
        dbc.Col([    
            dash.page_container
        ], xs=8,sm=8,md=10,lg=10,xl=10,xxl=10 )
    ])
     
],fluid=True)





clientside_callback = (
    """
    (switchOn)=>{
        document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
        return window.dash_clientside.no_update;
    }
    """,
    Output("color_mode_switch", "id"),
    Input("color_mode_switch", "value"),
)


if __name__ == "__main__":
    app.run_server(debug=True)
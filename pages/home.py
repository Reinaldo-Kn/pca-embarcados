from dash import html
import dash

dash.register_page(__name__,path="/",name="Sobre o projeto")

layout = html.H1("Sobre o projeto")
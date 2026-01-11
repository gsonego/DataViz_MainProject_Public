from shiny.express import input, ui, render
from shiny import reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shinywidgets import render_plotly
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import os
import warnings

# suppress joblib CPU detection warnings on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

# --- Loading data ---

# cleaned dataset
data_path = Path(__file__).parent.parent / "data" / "BRAZIL_CITIES_CLEANED.csv"
df = pd.read_csv(data_path)

# geojson for map boundaries
geojson_path = Path(__file__).parent.parent / "data" / "brazil-states.geojson"
with open(geojson_path, 'r', encoding='utf-8') as f:
    brazil_states_geojson = json.load(f)

# region colors for the map
region_fill_colors = {
    "1": "#f7ad5e",  # South
    "2": "#cfcfcf",  # Southeast
    "3": "#aa9e84",  # North
    "4": "#a9c3d1",  # Northeast
    "5": "#033234",  # MidWest
}

features_by_region = {}
for feature in brazil_states_geojson["features"]:
    rid = feature["properties"].get("regiao_id")
    if not rid:
        continue
    features_by_region.setdefault(rid, []).append(feature)

# min/max values for filter sliders
min_pop = int(df['ESTIMATED_POP'].min())
max_pop = int(df['ESTIMATED_POP'].max())

# unique regions and states
regions = sorted(df['REGION'].unique().tolist())
states = sorted(df['STATE'].unique().tolist())

# HDI and GDP ranges
min_hdi = round(df['IDHM'].min(), 3)
max_hdi = round(df['IDHM'].max(), 3)
min_gdp = int(df['GDP_CAPITA'].min())
max_gdp = int(df['GDP_CAPITA'].max())

# --- Reactive functions ---

@reactive.calc
def filtered_data():
    """applies global filters to data"""
    filtered_df = df.copy()
    
    # region filter
    if input.region_filter():
        filtered_df = filtered_df[filtered_df['REGION'].isin(input.region_filter())]
    
    # population range
    pop_min, pop_max = input.population_filter()
    filtered_df = filtered_df[
        (filtered_df['ESTIMATED_POP'] >= pop_min) & 
        (filtered_df['ESTIMATED_POP'] <= pop_max)
    ]
    
    # HDI range
    hdi_min, hdi_max = input.hdi_filter()
    filtered_df = filtered_df[
        (filtered_df['IDHM'] >= hdi_min) & 
        (filtered_df['IDHM'] <= hdi_max)
    ]
    
    # GDP per capita range
    gdp_min, gdp_max = input.gdp_filter()
    filtered_df = filtered_df[
        (filtered_df['GDP_CAPITA'] >= gdp_min) & 
        (filtered_df['GDP_CAPITA'] <= gdp_max)
    ]
    
    # capital or non-capital cities
    city_types = input.city_type_filter()
    if city_types:
        capital_values = []
        if "capital" in city_types:
            capital_values.append(1)
        if "non_capital" in city_types:
            capital_values.append(0)
        filtered_df = filtered_df[filtered_df['CAPITAL'].isin(capital_values)]
    
    return filtered_df

@reactive.effect
@reactive.event(input.reset_filters)
def _reset_all_filters():
    """resets everything back to defaults"""
    ui.update_selectize("region_filter", selected=regions)
    ui.update_slider("population_filter", value=[min_pop, max_pop])
    ui.update_slider("hdi_filter", value=[min_hdi, max_hdi])
    ui.update_slider("gdp_filter", value=[min_gdp, max_gdp])
    ui.update_checkbox_group("city_type_filter", selected=["capital", "non_capital"])

# --- UI Layout ---

with ui.sidebar(width=400):
    ui.h4("🔍 Global Filters")
    ui.h6("ℹ️ Filters apply to all dashboard pages ", class_="text-muted")

    # status badge showing filter results
    with ui.card(class_="bg-info text-white"):
        @render.text
        def filter_status():
            data = filtered_data()
            total = len(df)
            filtered = len(data)
            percentage = (filtered / total * 100) if total > 0 else 0
            return f"📊 Showing {filtered:,} / {total:,} cities ({percentage:.1f}%)"
            
    
    # region dropdown
    ui.input_selectize(
        id="region_filter",
        label="Regions:",
        choices=regions,
        selected=regions,
        multiple=True
    )
    
    ui.hr()
    
    # population slider
    ui.input_slider(
        id="population_filter",
        label="Population Range:",
        min=min_pop,
        max=max_pop,
        value=[min_pop, max_pop],
        step=10000
    )
    
    # HDI slider
    ui.input_slider(
        id="hdi_filter",
        label="HDI Range:",
        min=min_hdi,
        max=max_hdi,
        value=[min_hdi, max_hdi],
        step=0.01
    )
    
    # GDP per capita slider
    ui.input_slider(
        id="gdp_filter",
        label="GDP per Capita Range (R$):",
        min=min_gdp,
        max=max_gdp,
        value=[min_gdp, max_gdp],
        step=1000
    )
    
    ui.hr()
    
    # capital vs non-capital checkbox
    ui.input_checkbox_group(
        id="city_type_filter",
        label="City Type:",
        choices={
            "capital": "Capital Cities",
            "non_capital": "Non-Capital Cities"
        },
        selected=["capital", "non_capital"]
    )
    
    ui.hr()
    
    # reset button
    ui.input_action_button(
        id="reset_filters",
        label="Reset All Filters",
        class_="btn-warning btn-block"
    )

# Navigation tabs

# header with title and map button
with ui.layout_columns(col_widths=[10, 2], style="margin-bottom: 20px;"):
    # titles on the left
    with ui.div():
        ui.h1("Brazilian Cities Dashboard - Shiny for Python", style="margin-bottom: 0.25rem;")
        ui.h5("A comprehensive data visualization dashboard exploring Brazilian cities dataset", 
              class_="text-muted", style="margin-top: 10px;")
    
    # map button on the right
    with ui.div(style="display: flex; align-items: center; justify-content: flex-end;"):
        ui.input_action_button(
            id="show_brazil_map",
            label="🌎 Brazil Regions",
            style="margin-top: 1rem;"
        )

# toggle map visibility
map_visible = reactive.value(False)

@reactive.effect
@reactive.event(input.show_brazil_map)
def _toggle_map():
    """show/hide map"""
    map_visible.set(not map_visible())

@reactive.effect
@reactive.event(input.close_map)
def _close_map():
    """close map"""
    map_visible.set(False)

# map panel (shows conditionally)
@render.ui
def brazil_map_panel():
    if map_visible():
        return ui.TagList(
            ui.div(
                ui.div(
                    ui.h5("Brazil Regions Map", style="display: inline; margin: 0; padding: 10px;"),
                    ui.input_action_button(
                        id="close_map",
                        label="✖",
                        class_="btn-sm btn-danger",
                        style="float: right;"
                    ),
                    style="background-color: #f8f9fa; border-bottom: 2px solid #0dcaf0; padding: 10px;"
                ),
                ui.img(src="Brazil.png", style="width: 100%; max-width: 700px; display: block; margin: auto; padding: 20px;"),
                style="margin-bottom: 1rem; border: 2px solid #0dcaf0; border-radius: 5px; background-color: #fefdfb;"
            )
        )
    return ui.div()

# Tabs setup

with ui.navset_tab(id="tabs", selected="home"):
    
    # HOME tab
    
    with ui.nav_panel("Home", value="home"):
        ui.br()
        
        # welcome content
        with ui.layout_columns(col_widths=[7, 5]):
            # text on left
            with ui.div():
                ui.h2("Welcome to the Brazilian Cities Dashboard", style="color: #0d6efd; margin-bottom: 1rem;")
                
                ui.h4("About This Dashboard", style="margin-top: 1.5rem; color: #495057;")
                ui.p(
                    """This interactive dashboard provides comprehensive insights into Brazilian municipalities, 
                    covering demographic, economic, and social development indicators across all regions of Brazil. 
                    The dashboard enables data-driven exploration of urban and rural patterns, regional disparities, 
                    and development trends.""",
                    style="font-size: 1.05rem; line-height: 1.6;"
                )
                
                ui.h4("Data Source", style="margin-top: 1.5rem; color: #495057;")
                ui.p(
                    """The dataset includes official statistics from Brazilian government sources (IBGE, IPEA) 
                    covering 5,578 cities. Key metrics include population estimates, Human Development Index (HDI), 
                    GDP, infrastructure coverage, agricultural production, and demographic composition.""",
                    style="font-size: 1.05rem; line-height: 1.6;"
                )
                
                ui.h4("Interactive Features", style="margin-top: 1.5rem; color: #495057;")
                ui.p(
                    """Use the sidebar filters to customize your analysis by region, population range, HDI, GDP per capita, 
                    and city type (capital/non-capital). All visualizations update dynamically based on your selections, 
                    allowing focused exploration of specific segments of Brazilian cities.""",
                    style="font-size: 1.05rem; line-height: 1.6;"
                )
            
            # Right column: Brazil map
            with ui.div(style="display: flex; align-items: center; justify-content: center;"):
                ui.img(
                    src="Brazil.png", 
                    style="width: 100%; max-width: 450px; border: 2px solid #dee2e6; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
                )
        
        ui.br()
        ui.hr()
        
        # dashboard sections
        ui.h3("Dashboard Sections", style="color: #0d6efd; margin-bottom: 1rem;")
        
        with ui.layout_columns(col_widths=[6, 6]):
            # left side tabs
            with ui.div():
                with ui.card(style="border-left: 4px solid #dc3545;"):
                    ui.h5("🏠 Home", style="color: #dc3545;")
                    ui.p("This page.")

                with ui.card(style="border-left: 4px solid #0d6efd;"):
                    ui.h5("📊 Overview", style="color: #0d6efd;")
                    ui.p("Key performance indicators and interactive maps showcasing regional patterns across Brazil.")
                
                with ui.card(style="border-left: 4px solid #17a2b8;"):
                    ui.h5("👥 Demographics", style="color: #17a2b8;")
                    ui.p("Population distribution, age structure, and urban-rural composition analysis.")
                
                with ui.card(style="border-left: 4px solid #28a745;"):
                    ui.h5("💰 Economic Indicators", style="color: #28a745;")
                    ui.p("GDP composition, wealth distribution, and economic performance across regions and sectors.")
                
                with ui.card(style="border-left: 4px solid #ffc107;"):
                    ui.h5("📈 Human Development (HDI)", style="color: #ffc107;")
                    ui.p("HDI rankings, component analysis, and development potential assessments by region.")
            
            # Right column tabs
            with ui.div():
                with ui.card(style="border-left: 4px solid #6f42c1;"):
                    ui.h5("🏗️ Infrastructure & Services", style="color: #6f42c1;")
                    ui.p("Banking coverage, transportation networks, and essential service distribution.")

                with ui.card(style="border-left: 4px solid #fd7e14;"):
                    ui.h5("🌾 Agriculture", style="color: #fd7e14;")
                    ui.p("Agricultural land use, livestock production, and state-level farming patterns.")
                
                with ui.card(style="border-left: 4px solid #e83e8c;"):
                    ui.h5("🔬 Cluster Analysis", style="color: #e83e8c;")
                    ui.p("Machine learning clustering to identify similar city groups based on socioeconomic profiles.")
                
                with ui.card(style="border-left: 4px solid #20c997;"):
                    ui.h5("📉 Regression Analysis", style="color: #20c997;")
                    ui.p("Predictive modeling to understand factors influencing Human Development Index outcomes.")
                
                with ui.card(style="border-left: 4px solid #6c757d;"):
                    ui.h5("🗺️ Regional Structure & Inequality", style="color: #6c757d;")
                    ui.p("Compares state-level socioeconomic patterns and contrasts capital cities with non-capital municipalities.")
                
        
        ui.br()
        
        # getting started info
        with ui.card(style="background-color: #e7f3ff; border: 2px solid #0d6efd;"):
            ui.h4("🚀 Getting Started", style="color: #0d6efd;")
            ui.p(
                """Begin by exploring the _Overview_ tab for a high-level view of Brazilian cities. 
                Use the _sidebar filters_ on the left to narrow down your analysis. 
                Each tab offers specialized visualizations—hover over charts for detailed information and click on map points to explore individual cities.""",
                style="font-size: 1.05rem; line-height: 1.6; margin-bottom: 0;"
            )
    
    # OVERVIEW tab
    
    with ui.nav_panel("Overview", value="overview"):
        ui.br()
        ui.h3("Key Performance Indicators")
        ui.br()
        
        # KPIs
        with ui.layout_columns(col_widths=[3, 3, 3, 3]):
            
            # states count
            with ui.card(class_="text-center", style="border: 2px solid #dee2e6; background-color: #f8f9fa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"):
                ui.h5("States", class_="card-title", style="color: #495057; font-weight: 600;")
                @render.text
                def kpi_states():
                    data = filtered_data()
                    return f"{data['STATE'].nunique()}"
            
            # cities count
            with ui.card(class_="text-center", style="border: 2px solid #dee2e6; background-color: #f8f9fa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"):
                ui.h5("Cities", class_="card-title", style="color: #495057; font-weight: 600;")
                @render.text
                def kpi_cities():
                    data = filtered_data()
                    return f"{len(data):,}"
            
            # total pop
            with ui.card(class_="text-center", style="border: 2px solid #dee2e6; background-color: #f8f9fa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"):
                ui.h5("Total Population", class_="card-title", style="color: #495057; font-weight: 600;")
                @render.text
                def kpi_population():
                    data = filtered_data()
                    total_pop = data['ESTIMATED_POP'].sum()
                    return f"{total_pop/1_000_000:.1f}M"
            
            # KPI 4: HDI
            with ui.card(class_="text-center", style="border: 2px solid #dee2e6; background-color: #f8f9fa; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"):
                ui.h5("Median HDI", class_="card-title", style="color: #495057; font-weight: 600;")
                @render.text
                def kpi_hdi():
                    data = filtered_data()
                    median_hdi = data['IDHM'].median()
                    return f"{median_hdi:.3f}"
        
        ui.br()
        ui.hr()
        
        # visualizations side by side
        
        with ui.layout_columns(col_widths=[6, 6]):
            
            # interactive map on the left
            with ui.card():
                ui.card_header("Interactive Brazil Map - Contextual View")
                
                # Filters side by side
                with ui.layout_columns(col_widths=[6, 6]):
                # dropdown for variable selection
                    ui.input_select(
                        id="map_variable",
                        label="Select Variable to Display:",
                        choices={
                            "IDHM": "Human Development Index (HDI)",
                            "GDP_CAPITA": "GDP per Capita"
                        },
                        selected="IDHM"
                    )
                    
                    # Dropdown to select normalization level
                    ui.input_select(
                        id="map_normalization",
                        label="Normalize by:",
                        choices={
                            "region": "Region",
                            "state": "State"
                        },
                        selected="region"
                    )
                
                @render_plotly
                def brazil_map():
                    data = filtered_data().copy()
                    
                    # Get selected variable and normalization level
                    var = input.map_variable()
                    norm_level = input.map_normalization()
                    
                    # Variable labels
                    var_labels = {
                        "IDHM": "HDI",
                        "GDP_CAPITA": "GDP per Capita (R$)"
                    }
                    
                    # Normalize by region or state using percentile ranking
                    if norm_level == "region":
                        data['percentile'] = data.groupby('REGION')[var].rank(pct=True) * 100
                        norm_label = "Region"
                    else:  # state
                        data['percentile'] = data.groupby('STATE')[var].rank(pct=True) * 100
                        norm_label = "State"
                    
                    # Assign fixed percentile bins (0-20, 20-40, 40-60, 60-80, 80-100)
                    data['percentile_bin'] = pd.cut(
                        data['percentile'],
                        bins=[0, 20, 40, 60, 80, 100],
                        labels=['0-20', '20-40', '40-60', '60-80', '80-100'],
                        include_lowest=True
                    )
                    
                    # Create custom hover text with raw value + relative position
                    if var == "IDHM":
                        data['hover_text'] = data.apply(
                            lambda row: f"<b>{row['CITY']}</b><br>" +
                                       f"State: {row['STATE']}<br>" +
                                       f"Region: {row['REGION']}<br>" +
                                       f"<br><b>HDI: {row['IDHM']:.3f}</b><br>" +
                                       f"Percentile in {norm_label}: {row['percentile']:.1f}%<br>" +
                                       f"Bin: {row['percentile_bin']}",
                            axis=1
                        )
                    else:  # GDP_CAPITA
                        data['hover_text'] = data.apply(
                            lambda row: f"<b>{row['CITY']}</b><br>" +
                                       f"State: {row['STATE']}<br>" +
                                       f"Region: {row['REGION']}<br>" +
                                       f"<br><b>GDP per Capita: R$ {row['GDP_CAPITA']:,.2f}</b><br>" +
                                       f"Percentile in {norm_label}: {row['percentile']:.1f}%<br>" +
                                       f"Bin: {row['percentile_bin']}",
                            axis=1
                        )
                    
                    # Create scatter map with percentile coloring
                    fig = px.scatter_geo(
                        data,
                        lat='LAT',
                        lon='LONG',
                        color='percentile',
                        size='ESTIMATED_POP',
                        hover_name='hover_text',
                        color_continuous_scale='RdYlGn',  # Red-Yellow-Green scale
                        range_color=[0, 100],  # Fixed scale 0-100 percentile
                        title=f'Brazilian Cities - {var_labels[var]} (Normalized by {norm_label})',
                        size_max=15
                    )
                    
                    # Update hover template to show only the custom text
                    fig.update_traces(
                        hovertemplate='%{hovertext}<extra></extra>'
                    )
                    
                    # Add regional fills (soft colors) and state borders
                    for rid, feats in features_by_region.items():
                        fig.add_trace(go.Choropleth(
                            geojson={"type": "FeatureCollection", "features": feats},
                            locations=[f['properties']['sigla'] for f in feats],
                            z=[1] * len(feats),
                            featureidkey="properties.sigla",
                            showscale=False,
                            colorscale=[[0, region_fill_colors[rid]], [1, region_fill_colors[rid]]],
                            marker_line_color='black',
                            marker_line_width=0.8,
                            hoverinfo='skip'
                        ))
                    # Thin boundary overlay for all states
                    fig.add_trace(go.Choropleth(
                        geojson=brazil_states_geojson,
                        locations=[f['properties']['sigla'] for f in brazil_states_geojson['features']],
                        z=[1] * len(brazil_states_geojson['features']),
                        featureidkey="properties.sigla",
                        showscale=False,
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                        marker_line_color='black',
                        marker_line_width=1.0,
                        hoverinfo='skip'
                    ))
                    
                    # Update layout for Brazil
                    fig.update_geos(
                        scope='south america',
                        center=dict(lat=-14, lon=-55),
                        projection_scale=3.5,
                        showland=True,
                        landcolor='lightgray',
                        showlakes=True,
                        lakecolor='lightblue',
                        showcountries=True,
                        countrycolor='white',
                        fitbounds="locations"
                    )
                    
                    fig.update_layout(
                        height=600,
                        margin=dict(l=0, r=0, t=40, b=0),
                        coloraxis_colorbar=dict(
                            title="Percentile",
                            tickvals=[0, 20, 40, 60, 80, 100],
                            ticktext=['0', '20', '40', '60', '80', '100']
                        )
                    )
                    
                    return fig
            
            # Right: Regional Comparison Chart
            with ui.card():
                ui.card_header("Regional Comparison")
                
                @render_plotly
                def regional_comparison():
                    data = filtered_data()
                    
                    # Group by region and calculate averages (only IDHM and GDP_CAPITA)
                    regional_stats = data.groupby('REGION').agg({
                        'IDHM': 'mean',
                        'GDP_CAPITA': 'mean'
                    }).reset_index()
                    
                    # Sort by HDI (descending order)
                    regional_stats = regional_stats.sort_values('IDHM', ascending=False)
                    
                    # Create grouped bar chart with only 2 metrics
                    fig = go.Figure()
                    
                    # Add HDI bars (scaled x1000 for visibility)
                    fig.add_trace(go.Bar(
                        name='HDI (×1000)',
                        x=regional_stats['REGION'],
                        y=regional_stats['IDHM'] * 1000,
                        marker_color='steelblue',
                        text=[f"{val:.3f}" for val in regional_stats['IDHM']],
                        textposition='outside'
                    ))
                    
                    # Add GDP per Capita bars (scaled /100 for visibility)
                    fig.add_trace(go.Bar(
                        name='GDP per Capita (÷100 R$)',
                        x=regional_stats['REGION'],
                        y=regional_stats['GDP_CAPITA'] / 100,
                        marker_color='coral',
                        text=[f"{val:,.0f}" for val in regional_stats['GDP_CAPITA']],
                        textposition='outside'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Average HDI and GDP per Capita by Region',
                        xaxis_title='Region',
                        yaxis_title='Value (scaled)',
                        barmode='group',
                        height=600,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode='x unified'
                    )
                    
                    return fig
    
    # DEMOGRAPHICS tab
    
    with ui.nav_panel("Demographics", value="demographics"):
        ui.br()
        ui.h3("Demographic Analysis")
        ui.br()
        # POPULATION PYRAMID (Full Width)
        with ui.card():
            ui.card_header("Population Pyramid by Age Group")
            
            @render.plot
            def population_pyramid():
                data = filtered_data()
                
                # Age group columns
                age_columns = ['IBGE_1', 'IBGE_1-4', 'IBGE_5-9', 'IBGE_10-14', 'IBGE_15-59', 'IBGE_60+']
                age_labels = ['0-1', '1-4', '5-9', '10-14', '15-59', '60+']
                
                # Sum population by age group
                age_totals = [data[col].sum() for col in age_columns]
                
                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                
                y_pos = range(len(age_labels))
                bars = ax.barh(y_pos, age_totals, color='steelblue', edgecolor='black')
                
                # Customize
                ax.set_yticks(y_pos)
                ax.set_yticklabels(age_labels)
                ax.set_xlabel('Population', fontsize=10, fontweight='bold', color="#2e3d5e")
                ax.set_ylabel('Age Group', fontsize=10, fontweight='bold', color="#2e3d5e")
                ax.set_title('Population Distribution by Age Group', color="#2e3d5e", fontsize=12, pad=10)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, age_totals):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2, 
                           f' {value:,.0f}',
                           ha='left', va='center', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                return fig
        
        ui.br()
        # ROW 2: POPULATION DISTRIBUTION + URBAN VS RURAL (Side by Side)
        with ui.layout_columns(col_widths=[6, 6]):
            
            # LEFT: Population Distribution (Histogram + Box plot)
            with ui.card():
                ui.card_header("Population Distribution")
                
                # Toggle for log scale
                ui.input_switch(
                    id="pop_dist_log_scale",
                    label="Use Logarithmic Scale",
                    value=True  # Enabled by default for better visualization
                )
                
                @render_plotly
                def population_distribution():
                    data = filtered_data()
                    use_log = input.pop_dist_log_scale()
                    
                    # Create subplots: histogram on top, box plot below
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('Histogram', 'Box Plot'),
                        vertical_spacing=0.15
                    )
                    
                    # Histogram - create bins appropriately for scale
                    if use_log:
                        # For log scale, create logarithmically spaced bins
                        import numpy as np
                        pop_data = data['ESTIMATED_POP'].values
                        
                        # Filter out zero or negative values for log scale
                        pop_data = pop_data[pop_data > 0]
                        
                        if len(pop_data) > 0:
                            log_bins = np.logspace(np.log10(pop_data.min()), np.log10(pop_data.max()), 60)
                            
                            hist_counts, bin_edges = np.histogram(pop_data, bins=log_bins)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            bin_widths = bin_edges[1:] - bin_edges[:-1]
                            
                            fig.add_trace(
                                go.Bar(
                                    x=bin_centers,
                                    y=hist_counts,
                                    width=bin_widths * 0.9,  # Use 90% of bin width
                                    name='Population',
                                    marker_color='steelblue',
                                    marker_line_color='darkblue',
                                    marker_line_width=0.5,
                                    showlegend=False
                                ),
                                row=1, col=1
                            )
                        else:
                            # Fallback if no valid data
                            fig.add_trace(
                                go.Bar(x=[], y=[], marker_color='steelblue', showlegend=False),
                                row=1, col=1
                            )
                    else:
                        # Linear scale - use standard histogram with fewer bins for better visibility
                        fig.add_trace(
                            go.Histogram(
                                x=data['ESTIMATED_POP'],
                                nbinsx=30,  # Reduced bins for thicker bars
                                name='Population',
                                marker_color='steelblue',
                                marker_line_color='darkblue',
                                marker_line_width=0.5,
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    # Box plot
                    fig.add_trace(
                        go.Box(
                            x=data['ESTIMATED_POP'],
                            name='Population',
                            marker_color='coral',
                            showlegend=False,
                            boxpoints=False  # Remove individual outlier points
                        ),
                        row=2, col=1
                    )
                    
                    # Update axes
                    xaxis_type = 'log' if use_log else 'linear'
                    
                    fig.update_xaxes(title_text="Population", type=xaxis_type, row=1, col=1)
                    fig.update_xaxes(title_text="Population", type=xaxis_type, row=2, col=1)
                    fig.update_yaxes(title_text="Frequency", row=1, col=1)
                    
                    fig.update_layout(
                        height=500,
                        title_text=f"Population Distribution ({'Log' if use_log else 'Linear'} Scale)",
                        showlegend=False
                    )
                    
                    return fig
            
            # RIGHT: Urban vs Rural Analysis (Stacked Bar by Region)
            with ui.card():
                ui.card_header("Urban vs Rural Distribution by Region")
                
                @render_plotly
                def urban_rural_analysis():
                    data = filtered_data()
                    
                    # Group by region and sum domestic units
                    regional_urban_rural = data.groupby('REGION').agg({
                        'IBGE_DU_URBAN': 'sum',
                        'IBGE_DU_RURAL': 'sum'
                    }).reset_index()
                    
                    # Calculate total and sort by total population (ascending order)
                    regional_urban_rural['TOTAL'] = regional_urban_rural['IBGE_DU_URBAN'] + regional_urban_rural['IBGE_DU_RURAL']
                    regional_urban_rural = regional_urban_rural.sort_values('TOTAL', ascending=False)
                    
                    # Create stacked bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Urban',
                        x=regional_urban_rural['REGION'],
                        y=regional_urban_rural['IBGE_DU_URBAN'],
                        marker_color='lightblue',
                        text=regional_urban_rural['IBGE_DU_URBAN'],
                        texttemplate='%{text:,.0f}',
                        textposition='inside'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Rural',
                        x=regional_urban_rural['REGION'],
                        y=regional_urban_rural['IBGE_DU_RURAL'],
                        marker_color='lightgreen',
                        text=regional_urban_rural['IBGE_DU_RURAL'],
                        texttemplate='%{text:,.0f}',
                        textposition='inside'
                    ))
                    
                    fig.update_layout(
                        barmode='stack',
                        title='Domestic Units: Urban vs Rural',
                        xaxis_title='Region',
                        yaxis_title='Number of Domestic Units',
                        height=500,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        hovermode='x unified'
                    )
                    
                    return fig
        
        ui.br()
    
    # ECONOMIC INDICATORS tab
    
    with ui.nav_panel("Economic Indicators", value="economy"):
        ui.br()
        ui.h3("Economic Analysis")
        ui.br()
        # GDP COMPOSITION TREEMAP (Full Width)
        with ui.card():
            ui.card_header("GDP Composition by Sector")
            
            @render_plotly
            def gdp_composition_treemap():
                data = filtered_data()
                
                # sum up GVA by region, state and sector
                sectors = ['GVA_AGROPEC', 'GVA_INDUSTRY', 'GVA_SERVICES', 'GVA_PUBLIC']
                sector_names = ['Agriculture', 'Industry', 'Services', 'Public']
                
                # Prepare data for treemap with Region → State hierarchy
                treemap_data = []
                
                for region in data['REGION'].unique():
                    region_data = data[data['REGION'] == region]
                    
                    for state in region_data['STATE'].unique():
                        state_data = region_data[region_data['STATE'] == state]
                        
                        for sector, sector_name in zip(sectors, sector_names):
                            # Calculate total (sum) for absolute values
                            total_value = state_data[sector].sum()
                            treemap_data.append({
                                'Region': region,
                                'State': state,
                                'Sector': sector_name,
                                'Value': total_value,
                                'Label': f"{sector_name}<br>{state}"
                            })
                
                treemap_df = pd.DataFrame(treemap_data)
                
                # Create treemap with Region → State → Sector hierarchy
                fig = px.treemap(
                    treemap_df,
                    path=['Region', 'State', 'Sector'],
                    values='Value',
                    title='Economic Structure by Region and State (Total GVA by Sector)',
                    color='Value',
                    color_continuous_scale='RdYlGn',
                    hover_data={'Value': ':,.0f'}
                )
                
                fig.update_traces(
                    textinfo='label+value',
                    texttemplate='<b>%{label}</b><br>R$ %{value:,.0f}',
                    hovertemplate='<b>%{label}</b><br>Total Value: R$ %{value:,.0f}<extra></extra>'
                )
                
                fig.update_layout(
                    height=600,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                
                return fig
        
        ui.br()
        # ROW 2: GDP PER CAPITA ANALYSIS + WEALTH DISTRIBUTION (Side by Side)
        with ui.layout_columns(col_widths=[6, 6]):
            
            # LEFT: GDP per Capita Analysis (Scatter plot)
            with ui.card():
                ui.card_header("GDP per Capita vs Population")
                
                # Toggle for log scale
                ui.input_switch(
                    id="gdp_scatter_log_scale",
                    label="Use Logarithmic Scale (X-axis)",
                    value=True
                )
                
                @render_plotly
                def gdp_per_capita_analysis():
                    data = filtered_data()
                    use_log = input.gdp_scatter_log_scale()
                    
                    # Create scatter plot
                    fig = px.scatter(
                        data,
                        x='ESTIMATED_POP',
                        y='GDP_CAPITA',
                        color='REGION',
                        size='GDP',
                        hover_name='CITY',
                        hover_data={
                            'STATE': True,
                            'ESTIMATED_POP': ':,',
                            'GDP_CAPITA': ':,.2f',
                            'GDP': ':,.0f'
                        },
                        title='GDP per Capita vs Population by Region',
                        labels={
                            'ESTIMATED_POP': 'Population',
                            'GDP_CAPITA': 'GDP per Capita (R$)',
                            'REGION': 'Region'
                        },
                        log_x=use_log,
                        size_max=20
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title='Population' + (' (log scale)' if use_log else ''),
                        yaxis_title='GDP per Capita (R$)',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    return fig
            
            # RIGHT: Wealth Distribution (Violin plot)
            with ui.card():
                ui.card_header("Wealth Distribution by Region")
                
                @render_plotly
                def wealth_distribution():
                    data = filtered_data()
                    
                    # Create violin plot
                    fig = px.violin(
                        data,
                        y='GDP_CAPITA',
                        x='REGION',
                        color='REGION',
                        box=True,
                        points='outliers',
                        title='GDP per Capita Distribution by Region',
                        labels={
                            'GDP_CAPITA': 'GDP per Capita (R$)',
                            'REGION': 'Region'
                        },
                        hover_data={
                            'GDP_CAPITA': ':,.2f'
                        }
                    )
                    
                    fig.update_layout(
                        height=500,
                        showlegend=False,
                        xaxis_title='Region',
                        yaxis_title='GDP per Capita (R$)'
                    )
                    
                    fig.update_traces(
                        meanline_visible=True
                    )
                    
                    return fig
        
        ui.br()
    
    # HDI tab
    
    with ui.nav_panel("Human Development (HDI)", value="hdi"):
        ui.br()
        ui.h3("Human Development Index Analysis")
        ui.br()
        # ROW 1: HDI VS GDP CORRELATION + HDI IMPROVEMENT POTENTIAL (Side by Side)
        with ui.layout_columns(col_widths=[6, 6]):
            
            # LEFT: HDI vs GDP Correlation (Scatter with regression)
            with ui.card():
                ui.card_header("HDI vs GDP Correlation")
                
                # Toggle for log scale
                ui.input_switch(
                    id="hdi_gdp_log_scale",
                    label="Use Logarithmic Scale (X-axis)",
                    value=True
                )
                
                @render_plotly
                def hdi_gdp_correlation():
                    data = filtered_data()
                    use_log = input.hdi_gdp_log_scale()
                    
                    # Create scatter plot with trendline
                    fig = px.scatter(
                        data,
                        x='GDP_CAPITA',
                        y='IDHM',
                        color='REGION',
                        hover_name='CITY',
                        hover_data={
                            'STATE': True,
                            'GDP_CAPITA': ':,.2f',
                            'IDHM': ':.3f'
                        },
                        trendline='ols',
                        title='HDI vs GDP per Capita',
                        labels={
                            'GDP_CAPITA': 'GDP per Capita (R$)',
                            'IDHM': 'Human Development Index',
                            'REGION': 'Region'
                        },
                        log_x=use_log,
                        render_mode='svg'  # Force SVG rendering instead of WebGL
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title='GDP per Capita (R$)' + (' (log scale)' if use_log else ''),
                        yaxis_title='Human Development Index (HDI)',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    return fig
            
            # RIGHT: HDI Improvement Potential (Regional vs National Benchmarks)
            with ui.card():
                ui.card_header("HDI Improvement Potential by Region")
                
                # Benchmark selection
                ui.input_radio_buttons(
                    id="hdi_benchmark_type",
                    label="Benchmark Reference:",
                    choices={
                        "national_avg": "National Average",
                        "top_quartile": "Top Quartile (75th percentile)"
                    },
                    selected="national_avg",
                    inline=True
                )
                
                @render_plotly
                def hdi_improvement_potential():
                    data = filtered_data()
                    
                    # Calculate regional averages
                    regional_hdi = data.groupby('REGION').agg({
                        'IDHM_Renda': 'mean',
                        'IDHM_Longevidade': 'mean',
                        'IDHM_Educacao': 'mean',
                        'IDHM': 'mean'
                    }).reset_index()
                    
                    # Calculate national benchmarks
                    benchmark_type = input.hdi_benchmark_type()
                    if benchmark_type == "national_avg":
                        benchmarks = {
                            'IDHM_Renda': data['IDHM_Renda'].mean(),
                            'IDHM_Longevidade': data['IDHM_Longevidade'].mean(),
                            'IDHM_Educacao': data['IDHM_Educacao'].mean()
                        }
                        benchmark_label = "National Average"
                    else:  # top_quartile
                        benchmarks = {
                            'IDHM_Renda': data['IDHM_Renda'].quantile(0.75),
                            'IDHM_Longevidade': data['IDHM_Longevidade'].quantile(0.75),
                            'IDHM_Educacao': data['IDHM_Educacao'].quantile(0.75)
                        }
                        benchmark_label = "Top Quartile (75th %ile)"
                    
                    # Create bullet-style chart using grouped bars
                    fig = go.Figure()
                    
                    components = ['IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao']
                    component_names = ['Income', 'Longevity', 'Education']
                    colors = ['steelblue', 'coral', 'lightgreen']
                    
                    for comp, name, color in zip(components, component_names, colors):
                        fig.add_trace(go.Bar(
                            name=name,
                            x=regional_hdi['REGION'],
                            y=regional_hdi[comp],
                            marker_color=color,
                            text=[f"{val:.3f}" for val in regional_hdi[comp]],
                            textposition='outside'
                        ))
                    
                    # Add benchmark lines for each component
                    for comp, name, color in zip(components, component_names, colors):
                        fig.add_hline(
                            y=benchmarks[comp],
                            line_dash="dash",
                            line_color=color,
                            opacity=0.5,
                            annotation_text=f"{name}: {benchmarks[comp]:.3f}",
                            annotation_position="right",
                            annotation_font_size=10
                        )
                    
                    fig.update_layout(
                        title=f'HDI Components by Region vs {benchmark_label}',
                        xaxis_title='Region',
                        yaxis_title='HDI Component Value',
                        barmode='group',
                        height=500,
                        showlegend=True,
                        yaxis_range=[0, 1],
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    return fig
        
        ui.br()
        # HDI RANKING TABLE (Full Width)
        with ui.card():
            ui.card_header("HDI Ranking Table - Components Analysis")
            
            # Toggle for Top/Bottom
            ui.input_radio_buttons(
                id="hdi_ranking_filter",
                label="Show:",
                choices={"top": "Top 50", "bottom": "Bottom 50", "all": "All Cities"},
                selected="top",
                inline=True
            )
            
            @render.data_frame
            def hdi_ranking_table():
                data_all = filtered_data().copy()

                # Calculate percentile ranks on the full filtered dataset
                data_all['HDI_Percentile'] = data_all['IDHM'].rank(pct=True) * 100
                data_all['Income_Percentile'] = data_all['IDHM_Renda'].rank(pct=True) * 100
                data_all['Longevity_Percentile'] = data_all['IDHM_Longevidade'].rank(pct=True) * 100
                data_all['Education_Percentile'] = data_all['IDHM_Educacao'].rank(pct=True) * 100

                # Sort by HDI and assign ranking
                data_sorted = data_all.sort_values('IDHM', ascending=False).reset_index(drop=True)
                data_sorted['Ranking'] = range(1, len(data_sorted) + 1)

                # Apply filter (Top/Bottom/All)
                filter_type = input.hdi_ranking_filter()
                if filter_type == "top":
                    data_sorted = data_sorted.head(50)
                elif filter_type == "bottom":
                    data_sorted = data_sorted.tail(50)

                # Select and format columns - emphasis on HDI components and percentiles
                table_data = data_sorted[[
                    'Ranking', 'CITY', 'STATE', 'REGION',
                    'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao', 'IDHM',
                    'Income_Percentile', 'Longevity_Percentile', 'Education_Percentile', 'HDI_Percentile'
                ]].copy()
                
                # Rename columns for clarity
                table_data = table_data.rename(columns={
                    'IDHM_Renda': 'Income',
                    'IDHM_Longevidade': 'Longevity',
                    'IDHM_Educacao': 'Education',
                    'IDHM': 'HDI',
                    'Income_Percentile': 'Income %ile',
                    'Longevity_Percentile': 'Longevity %ile',
                    'Education_Percentile': 'Education %ile',
                    'HDI_Percentile': 'HDI %ile'
                })

                # Round numeric columns
                num_round_3 = ['Income', 'Longevity', 'Education', 'HDI']
                for col in num_round_3:
                    table_data[col] = table_data[col].round(3)

                # Percentiles as whole numbers
                pct_cols = ['Income %ile', 'Longevity %ile', 'Education %ile', 'HDI %ile']
                for col in pct_cols:
                    table_data[col] = table_data[col].round(0).astype(int)

                return render.DataTable(
                    table_data,
                    width="100%",
                    height="550px",
                    filters=True
                )
        
        ui.br()
    
    # INFRASTRUCTURE tab
    
    with ui.nav_panel("Infrastructure & Services", value="infrastructure"):
        ui.br()
        ui.h3("Infrastructure & Services Analysis")
        ui.br()
        # ROW 1: BANKING COVERAGE (Full Width)
        with ui.card():
            ui.card_header("Banking Coverage by Region")
            
            @render_plotly
            def banking_coverage():
                data = filtered_data()
                
                # Group by region and sum banking metrics
                regional_banking = data.groupby('REGION').agg({
                    'Pr_Agencies': 'sum',
                    'Pu_Agencies': 'sum',
                    'Pr_Bank': 'sum',
                    'Pu_Bank': 'sum'
                }).reset_index()
                
                # Sort by region name
                regional_banking = regional_banking.sort_values('REGION')
                
                # Create grouped bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Private Agencies',
                    x=regional_banking['REGION'],
                    y=regional_banking['Pr_Agencies'],
                    marker_color='steelblue',
                    text=regional_banking['Pr_Agencies'],
                    texttemplate='%{text:,}',
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Public Agencies',
                    x=regional_banking['REGION'],
                    y=regional_banking['Pu_Agencies'],
                    marker_color='coral',
                    text=regional_banking['Pu_Agencies'],
                    texttemplate='%{text:,}',
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Private Banks',
                    x=regional_banking['REGION'],
                    y=regional_banking['Pr_Bank'],
                    marker_color='lightgreen',
                    text=regional_banking['Pr_Bank'],
                    texttemplate='%{text:,}',
                    textposition='outside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Public Banks',
                    x=regional_banking['REGION'],
                    y=regional_banking['Pu_Bank'],
                    marker_color='gold',
                    text=regional_banking['Pu_Bank'],
                    texttemplate='%{text:,}',
                    textposition='outside'
                ))
                
                fig.update_layout(
                    barmode='group',
                    title='Banking Infrastructure by Region',
                    xaxis_title='Region',
                    yaxis_title='Count',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig
        
        ui.br()
        # ROW 2: TRANSPORTATION ASSETS (Full Width)
        with ui.card():
            ui.card_header("Transportation Assets by Region")
            
            @render_plotly
            def transportation_assets():
                data = filtered_data()
                
                # Group by region and calculate per capita
                regional_transport = data.groupby('REGION').agg({
                    'Cars': 'sum',
                    'Motorcycles': 'sum',
                    'ESTIMATED_POP': 'sum'
                }).reset_index()
                
                # Calculate per 1000 inhabitants
                regional_transport['Cars_per_1000'] = (regional_transport['Cars'] / regional_transport['ESTIMATED_POP']) * 1000
                regional_transport['Motorcycles_per_1000'] = (regional_transport['Motorcycles'] / regional_transport['ESTIMATED_POP']) * 1000
                
                # Sort by total vehicles per 1000
                regional_transport['Total_per_1000'] = regional_transport['Cars_per_1000'] + regional_transport['Motorcycles_per_1000']
                regional_transport = regional_transport.sort_values('Total_per_1000', ascending=True)
                
                # Create stacked bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Cars',
                    x=regional_transport['REGION'],
                    y=regional_transport['Cars_per_1000'],
                    marker_color='steelblue',
                    text=[f"{val:.1f}" for val in regional_transport['Cars_per_1000']],
                    textposition='inside'
                ))
                
                fig.add_trace(go.Bar(
                    name='Motorcycles',
                    x=regional_transport['REGION'],
                    y=regional_transport['Motorcycles_per_1000'],
                    marker_color='coral',
                    text=[f"{val:.1f}" for val in regional_transport['Motorcycles_per_1000']],
                    textposition='inside'
                ))
                
                fig.update_layout(
                    barmode='stack',
                    title='Vehicles per 1000 Inhabitants',
                    xaxis_title='Region',
                    yaxis_title='Vehicles per 1000 inhabitants',
                    height=500,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig
        
        ui.br()
    
    # AGRICULTURE tab
    
    with ui.nav_panel("Agriculture", value="agriculture"):
        ui.br()
        ui.h3("Agricultural Analysis")
        ui.br()
        # AGRICULTURAL LAND USE (Scatter plot - Full Width)
        with ui.card():
            ui.card_header("Agricultural Land Use & Productivity by State")
            
            @render_plotly
            def agricultural_land_use():
                data = filtered_data()
                
                # Group by state and aggregate agricultural data
                agro_data = data[(data['IBGE_PLANTED_AREA'] > 0) & (data['IBGE_CROP_PRODUCTION_$'] > 0)].copy()
                
                # grouping by state
                state_agro = agro_data.groupby('STATE').agg({
                    'IBGE_PLANTED_AREA': 'sum',
                    'IBGE_CROP_PRODUCTION_$': 'sum',
                    'REGION': 'first'  # Get the region for each state
                }).reset_index()
                
                # Calculate productivity ratio (production per hectare)
                state_agro['Productivity'] = state_agro['IBGE_CROP_PRODUCTION_$'] / state_agro['IBGE_PLANTED_AREA']
                
                # Handle infinite values
                state_agro['Productivity'] = state_agro['Productivity'].replace([float('inf'), float('-inf')], 0).fillna(0)
                
                # Create scatter plot
                fig = px.scatter(
                    state_agro,
                    x='IBGE_PLANTED_AREA',
                    y='IBGE_CROP_PRODUCTION_$',
                    color='REGION',
                    size='Productivity',
                    hover_name='STATE',
                    hover_data={
                        'REGION': True,
                        'IBGE_PLANTED_AREA': ':,',
                        'IBGE_CROP_PRODUCTION_$': ':,',
                        'Productivity': ':,.2f'
                    },
                    title='Agricultural Productivity by State: Planted Area vs Crop Production',
                    labels={
                        'IBGE_PLANTED_AREA': 'Planted Area (hectares)',
                        'IBGE_CROP_PRODUCTION_$': 'Crop Production Value (thousands R$)',
                        'REGION': 'Region',
                        'Productivity': 'Production per Hectare'
                    },
                    log_x=True,
                    log_y=True,
                    size_max=25
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        title="Region"
                    )
                )
                
                return fig
        
        ui.br()
        # ROW 2: AGRICULTURAL GDP CONTRIBUTION (Full Width)
        with ui.card():
            ui.card_header("Agricultural GDP Contribution")
            
            @render_plotly
            def agricultural_gdp_contribution():
                data = filtered_data()
                
                # Filter cities with valid data
                agro_gdp = data[(data['GVA_AGROPEC'] > 0) & (data['GVA_TOTAL'] > 0)].copy()
                
                # Calculate percentage contribution
                agro_gdp['Agro_Contribution_%'] = (agro_gdp['GVA_AGROPEC'] / agro_gdp['GVA_TOTAL']) * 100
                
                # Create scatter plot
                fig = px.scatter(
                    agro_gdp,
                    x='GVA_AGROPEC',
                    y='GVA_TOTAL',
                    color='REGION',
                    size='Agro_Contribution_%',
                    hover_name='CITY',
                    hover_data={
                        'STATE': True,
                        'GVA_AGROPEC': ':,.0f',
                        'GVA_TOTAL': ':,.0f',
                        'Agro_Contribution_%': ':.1f'
                    },
                    title='Agriculture Contribution to Total GDP',
                    labels={
                        'GVA_AGROPEC': 'Agricultural GVA (R$ thousands)',
                        'GVA_TOTAL': 'Total GVA (R$ thousands)',
                        'REGION': 'Region',
                        'Agro_Contribution_%': 'Agriculture %'
                    },
                    log_x=True,
                    log_y=True,
                    size_max=30
                )
                
                # Add diagonal reference line (where agro = total, which is impossible but for reference)
                fig.add_shape(
                    type="line",
                    x0=agro_gdp['GVA_AGROPEC'].min(),
                    y0=agro_gdp['GVA_AGROPEC'].min(),
                    x1=agro_gdp['GVA_AGROPEC'].max(),
                    y1=agro_gdp['GVA_AGROPEC'].max(),
                    line=dict(color="gray", width=1, dash="dash"),
                )
                
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                return fig
        
        ui.br()
    
    # CLUSTER ANALYSIS tab
    
    with ui.nav_panel("Cluster Analysis", value="cluster"):
        ui.br()
        ui.h3("PCA & Clustering Analysis")
        ui.br()
        
        # Number of clusters selector
        ui.input_slider(
            id="n_clusters",
            label="Number of Clusters (k):",
            min=3,
            max=8,
            value=5,
            step=1
        )
        
        ui.br()
        # PCA SCATTER PLOT (Full Width)
        with ui.card():
            ui.card_header("PCA Scatter Plot with Cluster Assignment")
            
            @render_plotly
            def pca_scatter_plot():
                data = filtered_data()
                
                # Select features for PCA
                features = ['ESTIMATED_POP', 'GDP_CAPITA', 'IDHM', 'IDHM_Renda', 'IDHM_Longevidade', 
                           'IDHM_Educacao', 'GVA_AGROPEC', 'GVA_INDUSTRY', 'GVA_SERVICES', 'AREA']
                
                # Prepare data (drop NaN and infinite values)
                pca_data = data[features + ['CITY', 'STATE', 'REGION', 'LAT', 'LONG']].copy()
                pca_data = pca_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(pca_data) < input.n_clusters():
                    # Not enough data
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Not enough data points for clustering. Try adjusting filters.",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=16)
                    )
                    return fig
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(pca_data[features])
                
                # Apply PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=input.n_clusters(), random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Add PCA components and clusters to dataframe
                pca_data['PC1'] = X_pca[:, 0]
                pca_data['PC2'] = X_pca[:, 1]
                pca_data['Cluster'] = clusters.astype(str)
                
                # Create scatter plot
                fig = px.scatter(
                    pca_data,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    hover_name='CITY',
                    hover_data={
                        'STATE': True,
                        'REGION': True,
                        'PC1': ':.2f',
                        'PC2': ':.2f'
                    },
                    title=f'PCA Scatter Plot with {input.n_clusters()} Clusters (Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%})',
                    labels={
                        'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                        'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                        'Cluster': 'Cluster'
                    }
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=True
                )
                
                return fig
        
        ui.br()
        # ROW 2: CLUSTER PROFILES (Full Width)
        with ui.card():
            ui.card_header("Cluster Profiles (Mean Feature Values)")
            
            @render_plotly
            def cluster_profiles():
                data = filtered_data()
                
                # Select features
                features = ['ESTIMATED_POP', 'GDP_CAPITA', 'IDHM', 'GVA_AGROPEC', 
                           'GVA_INDUSTRY', 'GVA_SERVICES']
                feature_labels = ['Population', 'GDP per Capita', 'HDI', 'Agro GVA', 
                                'Industry GVA', 'Services GVA']
                
                # Prepare data
                cluster_data = data[features].copy()
                cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(cluster_data) < input.n_clusters():
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Not enough data",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
                
                # Standardize and cluster
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(cluster_data)
                
                kmeans = KMeans(n_clusters=input.n_clusters(), random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                cluster_data['Cluster'] = clusters
                
                # Calculate mean values per cluster (normalized)
                cluster_means = cluster_data.groupby('Cluster')[features].mean()
                
                # Normalize for better visualization (0-100 scale)
                from sklearn.preprocessing import MinMaxScaler
                norm_scaler = MinMaxScaler(feature_range=(0, 100))
                cluster_means_norm = pd.DataFrame(
                    norm_scaler.fit_transform(cluster_means.T).T,
                    columns=feature_labels,
                    index=[f'Cluster {i}' for i in range(input.n_clusters())]
                )
                
                # Create heatmap
                fig = px.imshow(
                    cluster_means_norm,
                    labels=dict(x="Feature", y="Cluster", color="Score"),
                    x=feature_labels,
                    y=cluster_means_norm.index,
                    color_continuous_scale='RdYlGn',
                    title='Cluster Profiles (Normalized 0-100)',
                    aspect='auto'
                )
                
                fig.update_layout(
                    height=500
                )
                
                return fig
        
        ui.br()
        # CLUSTER MAP (Geographic distribution - Full Width)
        with ui.card():
            ui.card_header("Cluster Geographic Distribution")
            
            @render_plotly
            def cluster_map():
                data = filtered_data()
                
                # Select features
                features = ['ESTIMATED_POP', 'GDP_CAPITA', 'IDHM', 'GVA_AGROPEC', 
                           'GVA_INDUSTRY', 'GVA_SERVICES', 'AREA']
                
                # Prepare data
                map_data = data[features + ['CITY', 'STATE', 'REGION', 'LAT', 'LONG']].copy()
                map_data = map_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(map_data) < input.n_clusters():
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Not enough data for clustering",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    return fig
                
                # Standardize and cluster
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(map_data[features])
                
                kmeans = KMeans(n_clusters=input.n_clusters(), random_state=42, n_init=10)
                map_data['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
                
                # Create scatter geo map
                fig = px.scatter_geo(
                    map_data,
                    lat='LAT',
                    lon='LONG',
                    color='Cluster',
                    hover_name='CITY',
                    hover_data={
                        'STATE': True,
                        'REGION': True,
                        'ESTIMATED_POP': ':,',
                        'GDP_CAPITA': ':,.2f',
                        'IDHM': ':.3f',
                        'LAT': False,
                        'LONG': False
                    },
                    title=f'Spatial Distribution of {input.n_clusters()} City Clusters',
                    size='ESTIMATED_POP',
                    size_max=15
                )
                
                # Add regional fills (soft colors) and state borders
                for rid, feats in features_by_region.items():
                    fig.add_trace(go.Choropleth(
                        geojson={"type": "FeatureCollection", "features": feats},
                        locations=[f['properties']['sigla'] for f in feats],
                        z=[1] * len(feats),
                        featureidkey="properties.sigla",
                        showscale=False,
                        colorscale=[[0, region_fill_colors[rid]], [1, region_fill_colors[rid]]],
                        marker_line_color='black',
                            marker_line_width=0.8,
                        hoverinfo='skip'
                    ))
                fig.add_trace(go.Choropleth(
                    geojson=brazil_states_geojson,
                    locations=[f['properties']['sigla'] for f in brazil_states_geojson['features']],
                    z=[1] * len(brazil_states_geojson['features']),
                    featureidkey="properties.sigla",
                    showscale=False,
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                    marker_line_color='black',
                    marker_line_width=1.0,
                    hoverinfo='skip'
                ))
                
                # Update layout for Brazil
                fig.update_geos(
                    scope='south america',
                    center=dict(lat=-14, lon=-55),
                    projection_scale=3.5,
                    showland=True,
                    landcolor='lightgray',
                    showlakes=True,
                    lakecolor='lightblue',
                    showcountries=True,
                    countrycolor='white',
                    fitbounds="locations"
                )
                
                fig.update_layout(
                    height=600,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                return fig
        
        ui.br()
    
    # REGRESSION ANALYSIS tab
    
    with ui.nav_panel("Regression Analysis", value="regression"):
        ui.br()
        ui.h3("Predictive Model: HDI from Economic & Demographic Variables")
        ui.br()
        # ROW 1: FEATURE IMPORTANCE + ACTUAL VS PREDICTED (Side by Side)
        with ui.layout_columns(col_widths=[5, 7]):
            
            # LEFT: Feature Importance
            with ui.card():
                ui.card_header("Feature Importance")
                
                @render_plotly
                def feature_importance():
                    data = filtered_data()
                    
                    # Predict IDHM
                    features = ['GDP_CAPITA', 'ESTIMATED_POP', 'GVA_AGROPEC', 'GVA_INDUSTRY', 
                               'GVA_SERVICES', 'AREA', 'IBGE_DU_URBAN']
                    target = 'IDHM'
                    title = 'Drivers of HDI'
                    
                    # Prepare data
                    model_data = data[features + [target]].copy()
                    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(model_data) < 50:
                        fig = go.Figure()
                        fig.add_annotation(
                            text="Not enough data for regression. Try adjusting filters.",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=14)
                        )
                        return fig
                    
                    # Train model
                    X = model_data[features]
                    y = model_data[target]
                    
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X, y)
                    
                    # Get feature importances
                    importances = pd.DataFrame({
                        'Feature': features,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=True)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        importances,
                        y='Feature',
                        x='Importance',
                        orientation='h',
                        title=title,
                        labels={'Importance': 'Importance Score', 'Feature': 'Variable'},
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=500,
                        showlegend=False
                    )
                    
                    return fig
            
            # RIGHT: Actual vs Predicted
            with ui.card():
                ui.card_header("Model Performance: Actual vs Predicted")
                
                @render_plotly
                def actual_vs_predicted():
                    data = filtered_data()
                    
                    # Predict IDHM
                    features = ['GDP_CAPITA', 'ESTIMATED_POP', 'GVA_AGROPEC', 'GVA_INDUSTRY', 
                               'GVA_SERVICES', 'AREA', 'IBGE_DU_URBAN']
                    target = 'IDHM'
                    target_label = 'HDI'
                    
                    # Prepare data
                    model_data = data[features + [target, 'CITY', 'REGION']].copy()
                    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(model_data) < 50:
                        fig = go.Figure()
                        fig.add_annotation(
                            text="Not enough data",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False
                        )
                        return fig
                    
                    # Train-test split
                    X = model_data[features]
                    y = model_data[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    rf_model.fit(X_train, y_train)
                    
                    # Predictions
                    y_pred = rf_model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Actual vs Predicted points
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        marker=dict(color='steelblue', size=8, opacity=0.6),
                        name='Predictions'
                    ))
                    
                    # Perfect prediction line (diagonal)
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Perfect Prediction'
                    ))
                    
                    fig.update_layout(
                        title=f'R²: {r2:.3f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}',
                        xaxis_title=f'Actual {target_label}',
                        yaxis_title=f'Predicted {target_label}',
                        height=500,
                        showlegend=True
                    )
                    
                    return fig
    
    # REGIONAL STRUCTURE tab
    
    with ui.nav_panel("Regional Structure & Inequality", value="geographic"):
        ui.br()
        ui.h3("Regional Structure & Inequality")
        ui.br()
        
        # (3D Elevation Map removed)
        # ROW 2: STATE AGGREGATIONS + CAPITAL COMPARISON (Side by Side)
        with ui.layout_columns(col_widths=[7, 5]):
            
            # LEFT: State-Level Aggregations (Bar chart)
            with ui.card():
                ui.card_header("State-Level Aggregations")
                
                # Metric selector
                ui.input_select(
                    id="state_metric",
                    label="Select Metric:",
                    choices={
                        "IDHM": "Average HDI",
                        "GDP_CAPITA": "Average GDP per Capita"
                    },
                    selected="IDHM"
                )
                
                @render_plotly
                def state_level_aggregations():
                    data = filtered_data()
                    metric = input.state_metric()
                    
                    # Calculate state averages
                    state_stats = data.groupby('STATE').agg({
                        metric: 'mean',
                        'CITY': 'count'
                    }).reset_index()
                    
                    state_stats.columns = ['STATE', 'Metric', 'Cities']
                    state_stats = state_stats.sort_values('Metric', ascending=True)
                    
                    metric_label = "HDI" if metric == "IDHM" else "GDP per Capita (R$)"
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        state_stats,
                        y='STATE',
                        x='Metric',
                        orientation='h',
                        title=f'Average {metric_label} by State',
                        labels={
                            'Metric': metric_label,
                            'STATE': 'State'
                        },
                        hover_data={'Cities': True},
                        color='Metric',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig.update_layout(
                        height=800,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    return fig
            
            # RIGHT: Capital vs Non-Capital Comparison
            with ui.card():
                ui.card_header("Capital vs Non-Capital Cities")
                
                @render_plotly
                def capital_comparison():
                    data = filtered_data()
                    
                    # Add label for capital status
                    comp_data = data[['CAPITAL', 'IDHM', 'GDP_CAPITA', 'ESTIMATED_POP']].copy()
                    comp_data = comp_data.replace([np.inf, -np.inf], np.nan).dropna()
                    comp_data['City Type'] = comp_data['CAPITAL'].map({1: 'Capital', 0: 'Non-Capital'})
                    
                    # Create subplots: violin plots for HDI, GDP, and Population
                    from plotly.subplots import make_subplots
                    
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=('HDI Distribution', 'GDP per Capita Distribution', 'Population Distribution'),
                        vertical_spacing=0.12
                    )
                    
                    # HDI violin plot
                    for city_type in ['Capital', 'Non-Capital']:
                        data_subset = comp_data[comp_data['City Type'] == city_type]
                        fig.add_trace(
                            go.Violin(
                                y=data_subset['IDHM'],
                                name=city_type,
                                box_visible=True,
                                meanline_visible=True,
                                showlegend=(True if city_type == 'Capital' else False)
                            ),
                            row=1, col=1
                        )
                    
                    # GDP per Capita violin plot
                    for city_type in ['Capital', 'Non-Capital']:
                        data_subset = comp_data[comp_data['City Type'] == city_type]
                        fig.add_trace(
                            go.Violin(
                                y=data_subset['GDP_CAPITA'],
                                name=city_type,
                                box_visible=True,
                                meanline_visible=True,
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    
                    # Population violin plot
                    for city_type in ['Capital', 'Non-Capital']:
                        data_subset = comp_data[comp_data['City Type'] == city_type]
                        fig.add_trace(
                            go.Violin(
                                y=data_subset['ESTIMATED_POP'],
                                name=city_type,
                                box_visible=True,
                                meanline_visible=True,
                                showlegend=False
                            ),
                            row=3, col=1
                        )
                    
                    fig.update_yaxes(title_text="HDI", row=1, col=1)
                    fig.update_yaxes(title_text="GDP per Capita (R$)", row=2, col=1)
                    fig.update_yaxes(title_text="Population", type="log", row=3, col=1)
                    
                    fig.update_layout(
                        height=800,
                        title_text="Capital vs Non-Capital Comparison",
                        showlegend=True
                    )
                    
                    return fig
        
        ui.br()
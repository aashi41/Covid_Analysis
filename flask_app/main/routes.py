# External library
from flask import render_template, request, Blueprint
from . import covid_info, model, my_dashboard
import test
# Standard library
from typing import List, Tuple

main = Blueprint('main', __name__)


@main.route("/")
@main.route("/home")
def home():
    radio = request.args.get('radio')
    if not radio or radio and int(radio) == 0:
        print('pie option')
        images = covid_info.getDiagram_top5_corona_affected_cnty_comparison('pie')
    else:
        images = covid_info.getDiagram_top5_corona_affected_cnty_comparison('bar')

    # Get country input from dropdown
    country = request.args.get('country')
    if not country:
        country = 'India'
    country_wise_image = covid_info.getDiagram_Country_analysis(country)

    # Get table details
    cnt_details = covid_info.get_world_total_count()
    #table_output = get_table_header_rows()

    # Get list of countries for drop down
    countries = covid_info.get_country_list()
    country_data = covid_info.get_countrywise_total_count(country)

    mortality_rate,recovery_rate = covid_info.getRateAnalysis(country)
    

    return render_template(
        'home.html',
        images=images,
        table_headers=cnt_details.keys(),
        table_rows=cnt_details.values(),
        countries=countries,
        countrywise_image=country_wise_image,
        country_data=country_data,
        mortality_rate=mortality_rate,
        recovery_rate=recovery_rate
    )


@main.route("/about")
def about():
    return render_template(
        'about.html',
        title='About'
    )

@main.route("/dashboard")
def dashboard():
    global_covid_html = my_dashboard.get_choropleth_dia()
    print(global_covid_html)
    return render_template(
        'dashboard.html',
        title='dashboard',
        global_covid_html=global_covid_html
    )


@main.route("/prediction")
def prediction():    
    model_prediction = model.ploy_predict()
    return render_template(
        'prediction.html',
        title='prediction',
        model_prediction=model_prediction
    )



def get_bar_charts() -> List:
    return [
        "b1.jpeg",
        "b2.jpeg",
        "b3.jpeg"
    ]


def get_pie_charts() -> List:
    return [
        "p1.jpeg",
        "p2.jpeg",
        "p3.jpeg"
    ]


def get_list_of_countries() -> List:
    return [
        "Zimbabwe",
        "Afghanistan",
        "India"
    ]


def get_table_header_rows() -> Tuple:
    return ["Active", "Recovered", "Deaths", "Confirmed"], [54456, 67387, 1928, 4578]


def get_country_wise_image(country: str) -> str:
    return "country_output.jpeg"


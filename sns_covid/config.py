output_directory = 'c:/Users/Alex/PycharmProjects/SNS_covid/data'
data_source_url = 'https://covid.ourworldindata.org/data/owid-covid-data.json'
date_column_name = 'date'
date_column_format = '%Y-%m-%d'
country_iso_code = 'GBR'
input_columns = ['new_deaths_smoothed', 'new_cases_smoothed', 'stringency_index',
                 'reproduction_rate']
output_column = 'new_deaths_smoothed'
fill_override = []

epochs = 1000
batch_size = 16
n_input = 14
n_out = 7

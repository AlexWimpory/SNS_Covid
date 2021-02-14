output_directory = 'c:/Users/Alex/PycharmProjects/SNS_covid/data'
data_source_url = 'https://covid.ourworldindata.org/data/owid-covid-data.json'
date_column_name = 'date'
date_column_format = '%Y-%m-%d'
country_iso_code = 'GBR'
columns_used = ['new_deaths_smoothed', 'new_cases_smoothed', 'stringency_index', 'reproduction_rate']

epochs = 100
input_width = 7
shift = 1
label_columns = ['new_deaths_smoothed']
batch_size = 16
n_input = 14
n_out = 7

output_directory = './data'
owid_url = 'https://covid.ourworldindata.org/data/owid-covid-data.json'
gstatic_url = 'https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip'
gstatic_zip_name = '2020_{code}_Region_Mobility_Report.csv'
gstatic_filter = 'sub_region_1'
date_column_name = 'date'
date_column_format = '%Y-%m-%d'
country_iso_code = 'GBR'
country_alpha_2_code = 'GB'
input_columns = ['new_deaths_smoothed', 'new_cases_smoothed', 'stringency_index',
                 'reproduction_rate', 'retail_and_recreation_percent_change_from_baseline']
output_column = 'new_deaths_smoothed'
fill_override = []

epochs = 1000
batch_size = 16
n_input = 14
n_out = 7

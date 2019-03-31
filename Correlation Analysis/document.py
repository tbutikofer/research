import python.code as code, python.base as base

recalculate=False

securities = 100
simulation_runs = 1000
trading_days_list = [5, 10, 20, 40, 80, 160, 320]
change_range_uniform = (-0.5, 0.5)
baseline_stress_file = 'baseline_stress_{}'.format('uniform')
if recalculate:
	change_generator = base.uniform_distribution(change_range_uniform)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file, 20)
param_fit_uniform,  param_error_uniform = base.plot_baseline_stress(baseline_stress_file)

change_range_normal = (-0.05, 0.05)
baseline_stress_file = 'baseline_stress_{}'.format('normal')
sigma = 0.05
if recalculate:
	change_generator = base.normal_distribution(sigma, change_range_normal)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
param_fit_normal,  param_error_normal = base.plot_baseline_stress(baseline_stress_file)

######

market_code = 'SWX'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2000','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename = stress_series_file)
else:
	security_list = [None]*314
base.plot_time_series({'T=20':stress_series_file}, event_filename=event_filename)

data_range_zoom = ('01.01.2016','31.12.2016')
stress_series_zoom_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(data_range_zoom[0]), base.datestamp(data_range_zoom[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, data_range_zoom, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename = stress_series_zoom_file)
base.plot_time_series({'T=20':stress_series_zoom_file}, event_filename=event_filename)

#####

market_code = 'SWX'
date_range = ('01.01.2000','31.12.2018')
change_range = (-0.2, 0.2)
change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 314
market_param_SWX = (66, 123, 3.09, 110)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_SWX, r'$\Delta^{(SWX)}$')
#print(fit_param, fit_param_error)

baseline_stress_file = 'baseline_stress_{}'.format(market_code)
if recalculate:
	change_generator = base.market_distribution(
        market_param_SWX[0], market_param_SWX[1], market_param_SWX[2], market_param_SWX[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
market_fit_SWX,  market_fit_error_SWX = base.plot_baseline_stress(baseline_stress_file)
#print(market_fit_SWX, market_fit_error_SWX)
baseline_stress_20 = float(base.load_list('baseline_stress_SWX')[3,1])
#####

market_code = 'SWX'
event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
principal_securities_file = 'principal_securities_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	code.find_principal_securities(market_code, event_filename, date_range, series_length, principal_securities_file)
security_list = base.load_list(principal_securities_file)

#####

market_code = 'SWX'
series_length = 20
date_low = '18.03.2017'
stress_matrix_low_file = 'stress_{}_{}_{}'.format(market_code, series_length, base.datestamp(date_low))
if recalculate:
	code.market_stress_matrix(market_code, date_low, series_length, stress_matrix_low_file)
base.plot_matrix(stress_matrix_low_file, (-1,1))

date_high = '17.01.2015'
stress_matrix_high_file = 'stress_{}_{}_{}'.format(market_code, series_length, base.datestamp(date_high))
if recalculate:
	code.market_stress_matrix(market_code, date_high, series_length, stress_matrix_high_file)
base.plot_matrix(stress_matrix_high_file, (-1,1))

#####

market_code = 'XETR'
date_range = ('01.02.2003','31.12.2018')
change_range = (-0.2, 0.2)
change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 314
market_param_XETR = (56.8, 121, 3.04, 57)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_XETR, r'$\Delta^{(XETR)}$')
#print(fit_param, fit_param_error)

baseline_stress_file = 'baseline_stress_{}'.format(market_code)
if recalculate:
	change_generator = base.market_distribution(
        market_param_XETR[0], market_param_XETR[1], market_param_XETR[2], market_param_XETR[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
market_fit_XETR,  market_fit_error_XETR = base.plot_baseline_stress(baseline_stress_file)
#print(market_fit_XETR, market_fit_error_XETR)
baseline_stress_20 = float(base.load_list('baseline_stress_SWX')[3,1])

#####

market_code = 'XETR'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2003','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename=stress_series_file)
else:
	security_list = [None]*314
base.plot_time_series({'T=20':stress_series_file}, event_filename=event_filename)

#####

market_code = 'XETR'
event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
principal_securities_file = 'principal_securities_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	code.find_principal_securities(market_code, event_filename, date_range, series_length, principal_securities_file)
security_list = base.load_list(principal_securities_file)

#####
market_code = 'CRYP'
date_range = ('01.01.2015','31.12.2018')
change_range = (-0.2, 0.2)

change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 314
market_param_CRYP = (17.9, 84, 1.97, 0)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_CRYP, r'$\Delta^{(CRYP)}$')
#print(fit_param, fit_param_error)

baseline_stress_file = 'baseline_stress_{}'.format(market_code)
if recalculate:
	change_generator = base.market_distribution(
        market_param_CRYP[0], market_param_CRYP[1], market_param_CRYP[2], market_param_CRYP[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
market_fit_CRYP,  market_fit_error_CRYP = base.plot_baseline_stress(baseline_stress_file)
#print(market_fit_CRYP, market_fit_error_CRYP)
baseline_stress_20 = float(base.load_list('baseline_stress_SWX')[3,1])

#####

series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2015','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename=stress_series_file)
else:
	security_list = [None]*314
base.plot_time_series({'T=20':stress_series_file}, event_filename=event_filename)

securities = 100
simulation_runs = 1000
trading_days_list = [5, 10, 20, 40, 80, 160, 320]
baseline_stress_file = 'baseline_stress_synthetic'
file_list = [
	'baseline_stress_uniform','baseline_stress_normal',
	'baseline_stress_SWX','baseline_stress_XETR', 'baseline_stress_CRYP']
if False:
	synth_corr_param, synth_corr_error = code.baseline_stress_synthetic_correlation(file_list, trading_days_list, securities, simulation_runs, baseline_stress_file)
else:
	synth_corr_param = ([0.52310907, 0.35221981], [0.41051178, -1.22309267])
	synth_corr_error = ([0.01173427, 0.02156652], [0.01048636, 0.00392902])
market_fit_synth,  market_fit_error_synth = base.plot_baseline_stress(baseline_stress_file)
print(synth_corr_param, synth_corr_error)
print(market_fit_synth, market_fit_error_synth)

trading_days = 20
base.plot_correlation_histogram(trading_days, synth_corr_param, 'c')

#####

"""
change_range = (-0.05, 0.05)
data_generator = base.market_distribution(80, 120, 3.3, change_range)
data_generator = base.correlation_simulation_generator((0.52587211, 0.34470357, 0.40524949 , -1.46300512), 100, 20)
base.test_distribution(data_generator)
"""



import python.code as code, python.base as base

recalculate=False

# Introduction
# Methods
# Results
## Swiss stock exchange SWX
market_code = 'SWX'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2000','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename = stress_series_file)
else:
	security_list = [None]*246
base.plot_time_series({'T=20':stress_series_file},  (r'$f^{(SWX)}$'), event_filename=event_filename)

date_range_zoom = ('01.01.2016','31.12.2016')
stress_series_zoom_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range_zoom[0]), base.datestamp(date_range_zoom[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range_zoom, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename = stress_series_zoom_file)
base.plot_time_series({'T=20':stress_series_zoom_file}, (r'$f^{(SWX)}$'), event_filename=event_filename)



market_code = 'SWX'
date_range = ('01.01.2000','31.12.2018')
change_range = (-0.2, 0.2)
change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 473360
market_param_SWX = (66, 123, 3.09, 110)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_SWX, (r'$\Delta$',r'$p_{SWX}$'))
#print(fit_param, fit_param_error)


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



event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
principal_securities_file = 'principal_securities_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	code.find_principal_securities(market_code, event_filename, date_range, series_length, principal_securities_file)
security_list = base.load_list(principal_securities_file)


## German stock exchange XETR
market_code = 'XETR'
date_range = ('01.02.2003','31.12.2018')
change_range = (-0.2, 0.2)
change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, _ = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 619191
market_param_XETR = (56.8, 121, 3.04, 57)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_XETR, (r'$\Delta$',r'$p_{Xetra}$'))
#print(fit_param, fit_param_error)



market_code = 'XETR'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2003','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename=stress_series_file)
else:
	security_list = [None]*473

series_length_long = 40
stress_series_file_long = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length_long, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length_long)
	code.stress_series(change_generator, offset=0, filename=stress_series_file_long)
else:
	security_list = [None]*473

base.plot_time_series({'T=20':stress_series_file, 'T=40':stress_series_file_long}, (r'$f^{(Xetra)}$'), event_filename=event_filename)


market_code = 'XETR'
event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
principal_securities_file = 'principal_securities_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	code.find_principal_securities(market_code, event_filename, date_range, series_length, principal_securities_file)
security_list = base.load_list(principal_securities_file)


## Crypto currencies
market_code = 'CRYP'
date_range = ('01.01.2015','31.12.2018')
change_range = (-0.2, 0.2)
change_histogram_file = 'change_histogram_{}_{}-{}'.format(market_code, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range)
	sample_size = code.calculate_histogram(change_generator, change_range, change_histogram_file)
else:
	sample_size = 18990
market_param_CRYP = (17.9, 84, 1.97, 0)
fit_param, fit_param_error = base.plot_histogram(change_histogram_file, market_param_CRYP, (r'$\Delta$',r'$p_{CRYP}$'))
#print(fit_param, fit_param_error)



market_code = 'CRYP'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2015','31.12.2018')
stress_series_file = 'stress_series_{}_{}_{}-{}'.format(market_code, series_length, base.datestamp(date_range[0]), base.datestamp(date_range[1]))
if recalculate:
	change_generator, security_list = base.observed_changes(market_code, date_range, always=False, T=series_length)
	code.stress_series(change_generator, offset=0, filename=stress_series_file)
else:
	security_list = [None]*209
base.plot_time_series({'T=20':stress_series_file}, (r'$f^{(CRYP)}$'), event_filename=event_filename)

date_heat_map = '08.12.2018'
stress_matrix_file = 'stress_{}_{}_{}'.format(market_code, series_length, base.datestamp(date_heat_map))
if recalculate:
	code.market_stress_matrix(market_code, date_heat_map, series_length, stress_matrix_file)
base.plot_matrix(stress_matrix_file, (-1,1))


# Discussion
## Baseline market stres
securities = 100
simulation_runs = 1000
trading_days_list = [5, 10, 20, 40, 80, 160, 320]
correlation_check_T = 20

change_range_uniform = (-0.5, 0.5)
baseline_stress_file_uniform = 'baseline_stress_uniform'
if recalculate:
	change_generator = base.uniform_distribution(change_range_uniform)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file_uniform, correlation_check_T)
baseline_fit_uniform,  baseline_error_uniform = base.plot_baseline_stress(baseline_stress_file_uniform)
baseline_uniform_20 = float(base.load_list(baseline_stress_file_uniform)[3,1])

change_range_normal = (-0.05, 0.05)
baseline_stress_file = 'baseline_stress_normal'
sigma = 0.05
if recalculate:
	change_generator = base.normal_distribution(sigma, change_range_normal)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
baseline_fit_normal,  baseline_error_normal = base.plot_baseline_stress(baseline_stress_file)
baseline_normal_20 = float(base.load_list(baseline_stress_file)[3,1])

market_code = 'SWX'
change_range = (-0.1, 0.1)
baseline_stress_file = 'baseline_stress_SWX'
if recalculate:
	change_generator = base.market_distribution(
        market_param_SWX[0], market_param_SWX[1], market_param_SWX[2], market_param_SWX[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
baseline_fit_SWX,  baseline_error_SWX = base.plot_baseline_stress(baseline_stress_file)
baseline_SWX_20 = float(base.load_list(baseline_stress_file)[3,1])

market_code = 'XETR'
baseline_stress_file = 'baseline_stress_XETR'
if recalculate:
	change_generator = base.market_distribution(
        market_param_XETR[0], market_param_XETR[1], market_param_XETR[2], market_param_XETR[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
baseline_fit_XETR,  baseline_error_XETR = base.plot_baseline_stress(baseline_stress_file)
baseline_XETR_20 = float(base.load_list(baseline_stress_file)[3,1])

market_code = 'CRYP'
baseline_stress_file = 'baseline_stress_CRYP'
if recalculate:
	change_generator = base.market_distribution(
        market_param_CRYP[0], market_param_CRYP[1], market_param_CRYP[2], market_param_CRYP[3], change_range)
	code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, baseline_stress_file)
baseline_fit_CRYP,  baseline_error_CRYP = base.plot_baseline_stress(baseline_stress_file)
baseline_CRYP_20 = float(base.load_list(baseline_stress_file)[3,1])

baseline_stress_file = 'baseline_stress_synthetic'
file_fit_list = [
	'baseline_stress_uniform','baseline_stress_normal',
	'baseline_stress_SWX','baseline_stress_XETR', 'baseline_stress_CRYP']
if recalculate:
	synth_corr_param, synth_corr_error = code.baseline_stress_synthetic_correlation(file_fit_list, trading_days_list, securities, simulation_runs, baseline_stress_file)
else:
	synth_corr_param = ([0.52599842, 0.3481435], [0.41640685, -1.48280483])
	synth_corr_error = ([0.01103528, 0.02006066], [0.00095922, 0.13395941])
baseline_fit_synth,  baseline_error_synth = base.plot_baseline_stress(baseline_stress_file)
baseline_synth_20 = float(base.load_list(baseline_stress_file)[3,1])
base.plot_correlation_histogram(correlation_check_T, synth_corr_param, (r'$\chi$',r'$p(\chi)$'))


import python.code as code, python.base as base

"""
securities = 100
simulation_runs = 1000
trading_days_list = [5, 10, 20, 40, 80, 160, 320]
change_range = (-0.5, 0.5)
filename_uniform = 'market_baseline_stress_uniform'
change_generator = base.uniform_distribution(change_range)
code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, filename_uniform)
param_fit_uniform,  param_error_uniform = base.plot_baseline_stress(filename_uniform)

change_range = (-0.05, 0.05)
filename_normal = 'market_baseline_stress_normal'
sigma = 0.05
change_generator = base.normal_distribution(sigma, change_range)
code.baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, filename_normal)
param_fit_uniform,  param_error_uniform = base.plot_baseline_stress(filename_normal)
"""

"""
marketcode = 'SWX'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2000','31.12.2018')
#isin_count = base.market_info(marketcode, date_range)
market_stress_series = 'market_stress_series_SWX_20000101-20181231'
change_generator, security_list = base.observed_changes(marketcode, date_range, always=False, T=series_length)
code.stress_series(change_generator, offset=0, filename = market_stress_series)
base.plot_time_series(market_stress_series, {'T=20':market_stress_series}, event_filename=event_filename)

data_range_zoom = ('01.01.2016','31.12.2016')
market_stress_series_zoom = 'market_stress_series_SWX_20160101-20161231'
change_generator, _ = base.observed_changes(marketcode, data_range_zoom, always=False, T=series_length)
code.stress_series(change_generator, offset=0, filename = market_stress_series_zoom)
base.plot_time_series(market_stress_series_zoom, {'T=20':market_stress_series_zoom}, event_filename=event_filename)
"""

"""
marketcode = 'SWX'
date_range = ('01.01.2000','31.12.2018')
change_range = (-0.05, 0.05)
change_generator, _ = base.observed_changes(marketcode, date_range)
filename = 'change_histogram_SWX_20000101-20181231'
sample_size = code.calculate_histogram(change_generator, change_range, filename)
fit_param = (66, 152, 3.21)
base.plot_histogram(filename, fit_param, r'$\Delta^{(SWX)}$')
"""

"""
market_code = 'SWX'
event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
filename = 'market_base_SWX_20130101-20181231-20'
code.find_base_securities(market_code, event_filename, date_range, series_length, filename)
security_list = base.load_list(filename)
"""

"""
market_code = 'SWX'
series_length = 20

date_low = '18.03.2017'
filename_low = 'market_stress_SWX_20170318-20'
code.market_stress_matrix(market_code, date_low, series_length, filename_low)
base.plot_matrix(filename_low, (-1,1))

date_high = '17.01.2015'
filename_high = 'market_stress_SWX_20150117-20'
code.market_stress_matrix(market_code, date_high, series_length, filename_high)
base.plot_matrix(filename_high, (-1,1))
"""

"""
marketcode = 'XETR'
series_length = 20
event_filename = 'market_stress_events'

date_range = ('01.01.2003','31.12.2018')
#isin_count = base.market_info(marketcode, date_range)
market_stress_series = 'market_stress_series_XETR_20030101-20181231'
change_generator, security_list = base.observed_changes(marketcode, date_range, always=False, T=series_length)
code.stress_series(change_generator, offset=0, filename = market_stress_series)
base.plot_time_series(market_stress_series, {'T=20':market_stress_series}, event_filename=event_filename)
"""

"""
market_code = 'XETR'
event_filename = 'market_stress_events'
date_range = ('01.01.2013','31.12.2018')
series_length = 20
filename = 'market_base_XETR_20130101-20181231-20'
code.find_base_securities(market_code, event_filename, date_range, series_length, filename)
security_list = base.load_list(filename)
"""

"""
marketcode = 'XETR'
date_range = ('01.02.2003','31.12.2018')
change_range = (-0.05, 0.05)
change_generator, _ = base.observed_changes(marketcode, date_range)
filename = 'change_histogram_XETR_20030201-20181231'
sample_size = code.calculate_histogram(change_generator, change_range, filename)
fit_param = (56, 127, 3.11)
base.plot_histogram(filename, fit_param, r'$\Delta^{(XETR)}$')
"""

"""
marketcode = 'CRYP'
date_range = ('01.01.2015','31.12.2018')
change_range = (-0.2, 0.2)
change_generator, _ = base.observed_changes(marketcode, date_range)
filename = 'change_histogram_CRYP_20150101-20181231'
sample_size = code.calculate_histogram(change_generator, change_range, filename)
fit_param = (17.6, 81, 1.9)
base.plot_histogram(filename, fit_param, r'$\Delta^{(CRYP)}$')
"""

"""
change_range = (-0.05, 0.05)
data_generator = base.market_distribution(80, 120, 3.3, change_range)
data_generator = base.correlation_simulation_generator((0.52587211, 0.34470357, 0.40524949 , -1.46300512), 100, 20)
base.test_distribution(data_generator)
"""

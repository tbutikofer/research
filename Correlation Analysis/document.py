import python.code as code, python.base as base 

dataset_SWX_correlations = 'SWX_covariance_5_20_260-offset_0'
events_SWX = [
    ['0',  'Date', 'Event'],
    ['1',  '16.06.2006', 'unkown'],
    ['2',  '11.08.2007', 'EZB injects 150 bEUR due to US real estate and mortage crisis'],
    ['3',  '21.01.2008', 'Stock crash due to US real estate and mortage crisis'],
    ['4',  '13.10.2008', 'Speculations about EU bail-outs causes euphoria'],
    ['5',  '08.05.2010', 'EU 750 bEUR bail-out fonds / Greece crisis'],
    ['6',  '04.08.2011', 'EZB starts buying of government bonds'],
    ['7',  '06.09.2011', 'start of EUR:CHF ceiling'],
    ['8',  '15.01.2015', 'end of EUR:CHF ceiling'],
    ['9',  '24.08.2015', 'Mini-crash "black monday"'],
    ['10', '24.06.2016', 'Brexit referendum'],
    ['11', '05.02.2018', 'Mini-crash NYSE']
    ]


market_code = 'XTR'
date_from = '01.01.2003'
date_to = '31.12.2018'
series_length = 20

filename_pca ='market_pca_SWX_20160101-20161231_20'
#filename_pca = code.calc_PCA_market_dynamics(market_code,date_from,date_to, series_length=series_length, step_days=1, offset=0, num_components=3)
#base.plot_time_series(filename_pca, legends=None, events=None)

filename_momentum = 'market_momentum_SWX_20160101-20161231'
filename_momentum = code.calc_market_momentum(market_code,date_from,date_to, series_lengths=[series_length], step_days=7, offset=0)
base.plot_time_series(filename_momentum, legends=None, events=events_SWX)

#base.latex_table(events_SWX)

#filename = code.calc_market_stress('SWX', '18.03.2017', 20,1)
filename = 'market_stress_SWX_20150115-20'
#base.plot_matrix(filename, (-1,1))

#filename = code.find_base_securities('SWX', 'SWX_events', '01.01.2013', '31.12.2018', 20)
#filename = 'SWX_events'
#base.latex_table(filename)

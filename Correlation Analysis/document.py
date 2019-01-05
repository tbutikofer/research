import python.correlate as correlate, python.base as base 

dataset_SWX_correlations = 'SWX_covariance_5_20_260-offset_0'
events_SWX = [
    {'label':'1', 'date': '16.06.2006', 'text':'Acceptance of Slovenia a EU member state'},
    {'label':'2', 'date': '11.08.2007', 'text':'EZB injects 150 bEUR due to US real estate and mortage crisis'},
    {'label':'3', 'date': '21.01.2008', 'text':'Stock crash due to US real estate and mortage crisis'},
    {'label':'4', 'date': '13.10.2008', 'text':'Speculations about EU bail-outs causes euphoria'},
    {'label':'5', 'date': '08.05.2010', 'text':'EU 750 bEUR bail-out fonds / Greece crisis'},
    {'label':'6', 'date': '04.08.2011', 'text':'EZB starts buying of government bonds'},
    {'label':'7', 'date': '06.09.2011', 'text':'start of EUR:CHF ceiling'},
    {'label':'8', 'date': '15.01.2015', 'text':'end of EUR:CHF ceiling'},
    {'label':'9', 'date': '24.08.2015', 'text':'Mini-crash "black monday"'},
    {'label':'10', 'date': '24.06.2016', 'text':'Brexit referendum'},
    {'label':'11', 'date': '05.02.2018', 'text':'Mini-crash NYSE'},
    ]

#filename = correlate.calc_market_correlations('SWX', '01.01.2000', '31.12.2000', step_days=7, correlation_lengths=[5,20,260], offset=0)
#filename = 'market_correlation_SWX_20000101-20181231'
#correlate.plot_time_series(filename,['5 days','20 days','260 days'], events_SWX)

correlate.animate_market_dynamics('SWX','01.01.2000','31.12.2018', correlation_length=20, offset=0)

def plot_SWX_correlations(filename, events=None):
    timeseries = base.load_object(dataset_SWX_correlations)
    # correlate.plot_timeseries(timeseries['date'], timeseries['data'],
    #     ['5 days', '20 days', '260 days'],
    #     events,
    #     dataset_SWX_correlations)
#def plot_timeseries(date_series, value_series, legend_list, events=None, filename=None):

def plot_SWX_event(event):
    filename = "SWX_{}_28".format(base.date_stamp(event['date']))
    sample_data, date_series, security_num = correlate.sample_quote_correlations(
        marketcode='SWX',
        days=28,
        offset=0,
        today=event['date'],
        show_cluster=event['cluster'],
        sample_size=4,
        filename=None
        )
    return filename, sample_data, security_num

# filename, sample_data, security_num = plot_SWX_event(events_SWX[5])
# for sample_cluster in sample_data:
#     for data in sample_cluster['data']:
#         print("{} {} {} {:.3f}".format(sample_cluster['id'],data['id'],data['issuer'],float(data['norm'])))
# ii = 42
#base.dataset_to_csv(dataset_name)

#calc_SWX_correlations()
#plot_SWX_correlations()

#correlate.find_base_securities('SWX', events_SWX)

#calc_SWX_correlations()
#plot_SWX_correlations()


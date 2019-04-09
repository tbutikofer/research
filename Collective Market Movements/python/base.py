import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from datetime import datetime, timedelta, date
import pickle

DATE_FORMAT = '%d.%m.%Y'
DATESTAMP_FORMAT = '%Y%m%d'
LABEL_FONT_SIZE = 14

def get_db_connection():
    db_server = 'localhost'
    #    db_server = '192.168.0.42'
    db_user = 'moria_app'
    db_passwd = 'moria_app'
    db_scheme = 'moriadb'

    try:
        connection = MySQLdb.connect(db_server, db_user, db_passwd, db_scheme)
        connection.converter[MySQLdb.constants.FIELD_TYPE.DECIMAL] = float
        connection.converter[MySQLdb.constants.FIELD_TYPE.NEWDECIMAL] = float
    except:
        connection = None

    return connection

"""
def market_info(marketcode, date_range):
    db_connection = get_db_connection()
    query = "select count(distinct ts.isin) from t_market m \
            join t_traded_security ts on (m.marketid=ts.marketid) \
            join t_quote q on (ts.tradedsecurityid=q.tradedsecurityid) \
            where q.date between %(date_from)s and %(date_to)s \
            and m.marketcode=%(marketcode)s"
    
    df = pd.read_sql(query,
                        con=db_connection,
                        params={
                            'marketcode':marketcode,
                            'date_from':datetime.strptime(date_range[0], DATE_FORMAT),
                            'date_to':datetime.strptime(date_range[1], DATE_FORMAT)
                            })
    return df.values[0,0]
"""

def latex_table(data_array):
    row_template = " & ".join(["{}" for _ in data_array[0]])
    column_names = data_array[0]
    print(row_template.format(*column_names)+"\\\\")
    print(r"\hline")
    table_rows = []
    for table_row in data_array[1:]:
        table_row_esc = [value.replace('&', '\\&') for value in table_row]
        table_rows.append(row_template.format(*table_row_esc))
    table_string = "\\\\\n".join([row for row in table_rows])
    print(table_string)

def datestamp(date):
    return datetime.strptime(date, DATE_FORMAT).strftime(DATESTAMP_FORMAT)

def plot_time_series(filename_dict, axis_label, event_filename=None):
    style_list = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(11,6))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))

    i = 0
    date_range = [None, None]
    for label, filename in filename_dict.items():
        data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
        date_series = [datetime.strptime(date_str, DATE_FORMAT) for date_str in data_csv[1:,0]]
        data = data_csv[1:,1].astype(np.float)
        date_range[0] = np.min((date_range[0], date_series[0])) if date_range[0] else date_series[0]
        date_range[1] = np.max((date_range[1], date_series[-1])) if date_range[1] else date_series[-1]
        ax.plot(
            date_series,
            data,
            color='black',
            linestyle = style_list[i % len(style_list)],
            label=label
        )
        i = i + 1
    plt.ylabel(axis_label, fontsize=LABEL_FONT_SIZE)
    if event_filename:
        data_csv = np.loadtxt('csv/'+event_filename+'.csv', dtype=np.unicode, delimiter=',')
        date_series = [datetime.strptime(date_str, DATE_FORMAT) for date_str in data_csv[1:,1]]
        ylim = ax.get_ylim()
        for event in data_csv[1:,:]:
            date = datetime.strptime(event[1], DATE_FORMAT)
            plt.axvline(x=date, linestyle=':', color='black')
            plt.text(date,ylim[1]-(ylim[1]-ylim[0])/50.0,event[0],va='top')

    ax.set_xlim(date_range[0], date_range[1])
    fig.autofmt_xdate()
    ax.legend()
#    plt.show()
    fig.savefig('img/'+filename+".png", format='png')
    ax.legend().remove()
    plt.clf()

def plot_matrix(filename, val_range):
    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    data = data_csv[1:,1:].astype(float)
    
    fig, ax = plt.subplots(figsize=(6,6))

    img = plt.imshow(data, cmap=plt.get_cmap('jet'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.20)
    plt.clim(val_range)
    fig.colorbar(img, cax=cax)
    ax.axis('off')
    # plt.tight_layout(h_pad=1)
    fig.savefig('img/'+filename+".png", format='png')
    plt.clf()

def load_list(filename):
    return np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')

def plot_histogram(filename, func_param, axis_label):

    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    data = data_csv.astype(float)
    bin_avg = data[:,0]
    hist = data[:,1]
    bin_size = bin_avg[1]-bin_avg[0]

    plt.figure(figsize=(8,6))
    plt.bar(bin_avg, hist, width=bin_size, align='center', fill=False, log=True)

    popt, pcov = curve_fit(quote_change_distribution, bin_avg, hist, p0=func_param)
#    print(popt)
    plt.plot(bin_avg, quote_change_distribution(bin_avg, func_param[0], func_param[1], func_param[2], func_param[3]), 'k--', linewidth=2)
    plt.xlabel(axis_label[0], fontsize=LABEL_FONT_SIZE)
    plt.ylabel(axis_label[1], fontsize=LABEL_FONT_SIZE)
#    plt.show()
    plt.savefig('img/'+filename+".png", format='png')
    plt.clf()
    return (popt, np.sqrt(np.diag(pcov)))

def plot_correlation_histogram(T, func_param, axis_label):
    filename = 'correlation_histogram'
    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    data = data_csv[1:].astype(float)
    bin_avg = data[:,0]
    hist = data[:,1]
    bin_size = bin_avg[1]-bin_avg[0]

    plt.figure(figsize=(8,6))
    plt.bar(bin_avg, hist, width=bin_size, align='center', fill=False, log=True)
#    print(popt)
    plt.plot(bin_avg, correlation_distribution(bin_avg, np.power(T, func_param[0][0])*func_param[0][1], func_param[1][0]*T+func_param[1][1]), 'k--', linewidth=2)
    plt.xlim(-1,1)
    plt.xlabel(axis_label[0], fontsize=LABEL_FONT_SIZE)
    plt.ylabel(axis_label[1], fontsize=LABEL_FONT_SIZE)
#    plt.show()
    plt.savefig('img/'+filename+'.png', format='png')
    plt.clf()

def observed_changes(marketcode, date_range, always=True, T=1, isin_list=None):
    def read_raw_data(db_connection, marketcode, date_range, isin_list):
        """ Fetch quotes from database """
        query = "select i.name,ts.isin,q.date,q.quote from t_market m \
                join t_traded_security ts on m.marketid=ts.marketid \
                join t_quote q on ts.tradedsecurityid=q.tradedsecurityid \
                join t_security s on ts.isin=s.isin \
                join t_issuer i on s.issuerid=i.issuerid \
                where m.marketcode=%(marketcode)s \
                and q.date between %(from_date)s and %(to_date)s"
        if isin_list is not None:
            query = query + " and ts.isin in (" + ','.join('\'{0}\''.format(isin) for isin in isin_list) + ")"

        df = pd.read_sql(query,
                            con=db_connection,
                            params={
                                'marketcode':marketcode,
                                'from_date':datetime.strptime(date_range[0], DATE_FORMAT) - timedelta(3*T),
                                'to_date':datetime.strptime(date_range[1], DATE_FORMAT)
                                })
    #    df['date'] = df['date'].astype('datetime64[ns]')
        return df

    def observation_change_generator(data, data_date, T):
        i = T
        while True:
            if i<=len(data):
                yield data[i-T:i], data_date[i-1]
                i = i + 1
            else:
                raise StopIteration()

    db_connection = get_db_connection()
    df = read_raw_data(db_connection, marketcode, date_range, isin_list)
    df = df[df['quote'] > 0]
    df_isin_name = df[['isin','name']].drop_duplicates()
    isin_name_dict = {}
    for isin, name in zip(df_isin_name['isin'].values, df_isin_name['name'].values):
        isin_name_dict[isin] = name

    df = df.pivot(index='date', columns='isin', values='quote')
    if len(df) > 0:
        if always:
            # Remove isins without continuous time series
            first_day_quotes = df.iloc[0]
            last_day_quotes = df.iloc[-1]
            no_first_day_quote = first_day_quotes[first_day_quotes.isna()].index.values
            no_last_day_quote = last_day_quotes[last_day_quotes.isna()].index.values
            incomplete_series = list(set().union(no_first_day_quote, no_last_day_quote))
            df = df.drop(incomplete_series, axis=1)

        # Fill missing quotes with previous day quote
        df = df.fillna(method='pad')

    isin_list = np.array([isin for isin in df.columns])
    date_list = np.array([date for date in df.index])

    # Calculate relative quote changes (log)
    date_list = date_list[1:]
    data = df.values
    data_change = np.log(data[1:] / data[:-1])
    # remove days without trading (data_change = 0 for all securities)
    trading_days = ~np.all(data_change == 0, axis=1)
    data_change = data_change[trading_days,:]
    date_list = date_list[trading_days]

    date_start = datetime.strptime(date_range[0], DATE_FORMAT).date()
    date_start_idx = np.where(date_list<=date_start)[0][-1]
    date_list = date_list[date_start_idx-T+1:]
    data_change = data_change[date_start_idx-T+1:]

    # remove securities which haven't been traded
    securities_traded = ~np.all(data_change == 0, axis=0)
    data_change = data_change[:,securities_traded]

    isin_list = isin_list[securities_traded]
    security_list = np.array([[isin, isin_name_dict[isin]] for isin in isin_list])

    return observation_change_generator(data_change, date_list, T), security_list

def quote_change_simulation_generator(change_generator, securities, T, date_range=None):
    if date_range:
        date = datetime.strptime(date_range[0], DATE_FORMAT)
        date_end = datetime.strptime(date_range[1], DATE_FORMAT)
    else:
        date = None
        date_end = None
    
    change_buffer = []
    while (date_range is None) or (date<=date_end):
        quote_changes = []
        for _ in range(securities):
            x = next(change_generator)
            quote_changes.append(x)
        change_buffer.append(quote_changes)
        if len(change_buffer) >= T:
            yield np.array(change_buffer), date
            change_buffer.pop(0)
        if date:
            date = (np.datetime64(date) + np.timedelta64(1,'D')).astype(datetime)

def correlation_simulation_generator(dist_params, securities, T):
    prop_param = (
                    np.power(T, dist_params[0])*dist_params[1],
                    dist_params[2]*T + dist_params[3]
                )
    max_norm = -1.0
    for _ in range(10000):
        x = 2 * np.random.rand() - 1.0
        p = correlation_distribution(x, prop_param[0], prop_param[1])
        max_norm = max_norm if p<max_norm else p

    while True:
        corr = np.ones((securities, securities))
        for i in range(securities - 1):
            for j in range(i+1, securities):
                while True:
                    x = 2 * np.random.rand() - 1.0
                    acceptance_prop = np.random.rand()
                    p = correlation_distribution(x, prop_param[0], prop_param[1])
                    if p/max_norm > acceptance_prop:
                        break
                corr[i,j] = x
                corr[j,i] = x
        yield corr

def plot_baseline_stress(filename):
    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    data = data_csv[1:,:].astype(np.float)

    def func(x, a, b):
        return a * np.power(x,b)

    popt, pcov = curve_fit(func, data[:,0], data[:,1])
    #print(popt)
    #print(func(data[:,0], popt[0], popt[1]))

    color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    style_list = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(
        data[:,0],
        data[:,1],
        color='black',
        linestyle='None',
        marker='o',
        mfc='None'
        )
    ax.plot(
        data[:,0],
        func(data[:,0], popt[0], popt[1]),
        linestyle=':',
        color='black',
        )
    ax.axis([4, 350, 0.05, 0.8])
    plt.xscale('log')
    plt.xticks([5, 10, 20, 40, 80, 160, 320])
    plt.yscale('log')
    plt.yticks([0.05, 0.1, 0.2, 0.4, 0.8])
    plt.minorticks_off()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.floor(np.log10(y)),0))).format(y))))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.floor(np.log10(y)),0))).format(y))))
    fontsize = 18
    plt.xlabel('$T$', fontsize=fontsize)
    plt.ylabel('$f^0_T$', fontsize=fontsize)
    plt.savefig('img/'+filename+".png", format='png')
    plt.clf()

    return (popt, np.sqrt(np.diag(pcov)))

def test_distribution(data_generator):
    x = [next(data_generator) for _ in range(100000)]
    fig, ax = plt.subplots()

    # the histogram of the data
    num_bins = 100
    n, bins, patches = ax.hist(x, num_bins, density=1)
    plt.show()

def uniform_distribution(change_range):
    while True:
        x = np.random.uniform(change_range[0], change_range[1])
        yield x

def normal_distribution(sigma, change_range):
    while True:
        x = np.random.normal(scale=sigma)
        if change_range[0] <= x <= change_range[1]:
            yield x

def correlation_distribution(x, a, b):
    return a * np.power(np.abs(np.cos(x * np.pi / 2)), b)

def quote_change_distribution(x,a,b,c,d):
    return np.exp((a*x*(1-np.exp(b*x))/(1+np.exp(b*x))+c)*(1+np.exp(-d*x*x))/2)

def market_distribution(a, b, c, d, change_range):
    max_norm = -1.0
    for _ in range(10000):
        x = np.random.rand() * (change_range[1] - change_range[0]) + change_range[0]
        p = quote_change_distribution(x, a, b, c, d)
        max_norm = max_norm if p<max_norm else p

    while True:
        x = np.random.rand() * (change_range[1] - change_range[0]) + change_range[0]
        acceptance_prop = np.random.rand()
        p = quote_change_distribution(x, a, b, c, d) / max_norm
        if p > acceptance_prop:
            yield x

def format_number(number):
    return "{:,}".format(number).replace(",", "'")

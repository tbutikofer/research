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

def get_securities(marketcode):
    db_connection = get_db_connection()
    query = "select ts.isin, i.name from t_market m \
             join t_traded_security ts on m.marketid=ts.marketid \
             join t_security s on s.isin=ts.isin \
             join t_issuer i on s.issuerid=i.issuerid \
             where m.marketcode=%(marketcode)s"
    df = pd.read_sql(query,
                        con=db_connection,
                        params={'marketcode':marketcode})
    return np.array(df.values)

def read_raw_data(db_connection, marketcode, days, isin_list = None, today=date.today()):
    query = "select i.name,ts.isin,q.date,q.quote from t_market m \
	        join t_traded_security ts on m.marketid=ts.marketid \
            join t_quote q on ts.tradedsecurityid=q.tradedsecurityid \
            join t_security s on ts.isin=s.isin \
            join t_issuer i on s.issuerid=i.issuerid \
            where m.marketcode=%(marketcode)s \
            and q.date between (%(today)s - interval %(days)s day) and %(today)s"
    #            and q.date between (%(today)s - interval %(interval)d day) and %(today)s
    if isin_list is not None:
        query = query + " and ts.isin in (" + ','.join('\'{0}\''.format(isin) for isin in isin_list) + ")"

    df = pd.read_sql(query,
                        con=db_connection,
                        params={'marketcode':marketcode, 'today':today, 'days':days})
#    df['date'] = df['date'].astype('datetime64[ns]')
    return df

def market_info(marketcode, date_from, date_to):
    db_connection = get_db_connection()
    query = "select count(distinct ts.isin) from t_market m \
            join t_traded_security ts on (m.marketid=ts.marketid) \
            join t_quote q on (ts.tradedsecurityid=q.tradedsecurityid) \
            where q.date between %(date_from)s and %(date_to)s \
            and m.marketcode=%(marketcode)s"
    
    df = pd.read_sql(query,
                        con=db_connection,
                        params={'marketcode':marketcode, 'date_from':date_from, 'date_to':date_to})
    return df.values[0,0]


def cleanup_data(df):
    df = df.pivot(index='date', columns='isin', values='quote')
#    df.to_csv('quotes.csv')
    if len(df) > 0:

        # Remove isins without continuous time series
        first_day_quotes = df.iloc[0]
        last_day_quotes = df.iloc[-1]
    #    print(b.loc[:, b.isna().any()])
        no_first_day_quote = first_day_quotes[first_day_quotes.isna()].index.values
        no_last_day_quote = last_day_quotes[last_day_quotes.isna()].index.values
        incomplete_series = list(set().union(no_first_day_quote, no_last_day_quote))
        df = df.drop(incomplete_series, axis=1)

        # Fill missing quotes with previous day quote
        df = df.fillna(method='pad')
    return df

def read_buffer_data(db_connection, marketcode, isin_list = None):
    query = "select i.name,ts.isin,q.date,q.quote from t_market m \
	        join t_traded_security ts on m.marketid=ts.marketid \
            join t_quote q on ts.tradedsecurityid=q.tradedsecurityid \
            join t_security s on ts.isin=s.isin \
            join t_issuer i on s.issuerid=i.issuerid \
            where m.marketcode=%(marketcode)s "
    if isin_list is not None:
        query = query + " and ts.isin in (" + ','.join('\'{0}\''.format(isin) for isin in isin_list) + ")"
    df = pd.read_sql(query,
                        con=db_connection,
                        params={'marketcode':marketcode})
    
    df_isin_name = df[['isin','name']].drop_duplicates()
    isin_name = {}
    for isin, name in zip(df_isin_name['isin'].values, df_isin_name['name'].values):
        isin_name[isin] = name

    df = cleanup_data(df)
    return df, isin_name

def calc_correlation(df, offset=0):
    a = np.log(1 + df.pct_change()[1:])
    correlation = a.corr()
    return correlation

def get_clean_data(marketcode, days, isin_list = None, today_str=date.today().strftime(DATE_FORMAT)):
    today = datetime.strptime(today_str, DATE_FORMAT)

    # df = buffer_data[(today-timedelta(days=days)).date():today.date()]
    db_connection = get_db_connection()
    df = read_raw_data(db_connection, marketcode, days, isin_list, today)
    df = df[df['quote'] > 0]
    df_isin_name = df[['isin','name']].drop_duplicates()
    isin_name = {}
    for isin, name in zip(df_isin_name['isin'].values, df_isin_name['name'].values):
        isin_name[isin] = name

    df = cleanup_data(df)

    date_list = np.array([date for date in df.index])
    isin_list = np.array([isin for isin in df.columns])
    issuer_list = np.array([isin_name[isin] for isin in df.columns])
    return df.values, date_list, isin_list, issuer_list

def create_date_series(fromdate_str, todate_str, interval_days):
    date_series = []
    fromdate = datetime.strptime(fromdate_str, DATE_FORMAT)
    todate = datetime.strptime(todate_str, DATE_FORMAT)
    interval = timedelta(days=interval_days)
    date = fromdate
    while date <= todate:
        date_series.append(date)
        date += interval
    return date_series

def save_object(obj, filename ):
    with open('obj/'+ filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open('obj/' + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# def latex_table(column_names, alignment, values):
#     print(r"\begin{{tabular}}{{{}}}".format(" | ".join(alignment)))
#     row_template = " & ".join(["{}" for _ in column_names]) + "\\"
#     column_names = [list(column.keys())[0] for column in column_names]
#     print(row_template.format(*column_names) + "\\")
#     print(r"\hline")
#     for value in values:
#         print(r"{} & {} & {}\\".format(value['label'], value['date'], value['text']))
#     print(r"\end{tabular}")

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

def latex_table_old(table_array):
    row_template = " & ".join(["{}" for _ in table_array[0]])
    column_names = table_array[0]
    print(row_template.format(*column_names)+"\\\\")
    print(r"\hline")
    table_rows = []
    for table_row in table_array[1:]:
        table_row_esc = [value.replace('&', '\\&') for value in table_row]
        table_rows.append(row_template.format(*table_row_esc))
    table_string = "\\\\\n".join([row for row in table_rows])
    print(table_string)

def dataset_to_csv(dataset_name):
    timeseries = load_object(dataset_name)
    data = []
    header = [list(timeseries[0].keys())[0]]
    for column in list(timeseries[1].keys()):
        header.append(column)
    data.append(header)
    for i, date in enumerate(timeseries[0][header[0]]):
        row = [date.strftime(DATE_FORMAT)]
        for column in header[1:]:
            row.append(timeseries[1][column][i])
        data.append(row)
    np.savetxt(dataset_name+".csv", data, fmt='%s', delimiter=",")

def date_stamp(date_str):
    return datetime.strptime(date_str, DATE_FORMAT).strftime("%Y%m%d")
 
#buffer_data, isin_name = read_buffer_data(get_db_connection(), 'SWX')
ii=42

def plot_time_series(filename, legends, events):
    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    date_series = [datetime.strptime(date_str, DATE_FORMAT) for date_str in data_csv[1:,0]]
    if legends is None:
        legends = data_csv[0,1:]
    data = data_csv[1:,1:].astype(np.float)
    color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    style_list = ['-', '--', ':', '-.']
    fig, ax = plt.subplots(figsize=(11,6))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax.set_xlim(date_series[0], date_series[-1])
    fig.autofmt_xdate()

    for i, legend_txt in enumerate(legends):
        ax.plot(
            date_series,
            data[:,i],
            color=color_list[i % len(color_list)],
        #    color='black',
            linestyle = style_list[i % len(style_list)],
            label=legend_txt
            )

    ylim = ax.get_ylim()
    if events is not None:
        for event in events[1:]:
            date = datetime.strptime(event[1], DATE_FORMAT)
            plt.axvline(x=date, linestyle=':', color='black')
            plt.text(date,ylim[1]-(ylim[1]-ylim[0])/50.0,event[0],va='top')

    if not np.all([not a for a in legends]):
        legend = ax.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig('img/'+filename+".png", format='png')
    # legend = ax.legend().remove()
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

def quote_change_distribution(x, a, b, c):
    return np.exp(a*x*(1-np.exp(b*x))/(1+np.exp(b*x))+c)

def correlation_distribution(x, a, b):
    return a * np.power(np.abs(np.cos(x * np.pi / 2)), b)

def plot_histogram(filename, func_param, xlabel):

    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',')
    data = data_csv.astype(float)
    bin_avg = data[:,0]
    hist = data[:,1]
    bin_size = bin_avg[1]-bin_avg[0]

    plt.bar(bin_avg, hist, width=bin_size, align='center', fill=False, log=True)

    popt, pcov = curve_fit(quote_change_distribution, bin_avg, hist, p0=(22.0, 160.0, 2.4))
#    print(popt)
    plt.plot(bin_avg, quote_change_distribution(bin_avg, func_param[0], func_param[1], func_param[2]), 'k--', linewidth=2)
#    plt.show()
    plt.xlabel(xlabel)
    plt.savefig('img/'+filename+".png", format='png')
    plt.clf()

    return (popt, np.sqrt(np.diag(pcov)))

def quote_change_simulation_generator(dist_params, change_range):
    dist_type = len(dist_params)
    if dist_type == 3:
        a, b, c = dist_params
        max_norm = -1.0
        for _ in range(10000):
            x = np.random.rand() * (change_range[1] - change_range[0]) + change_range[0]
            p = quote_change_distribution(x, dist_params[0], dist_params[1], dist_params[2])
            max_norm = max_norm if p<max_norm else p
#    print(max_norm)

    while True:
        if dist_type == 0:
            x = np.random.rand() * (change_range[1] - change_range[0]) + change_range[0]
        elif dist_type == 1:
            x = np.random.randn()/np.sqrt(dist_params[0])
        elif dist_type == 3:
            while True:
                x = np.random.rand() * (change_range[1] - change_range[0]) + change_range[0]
                acceptance_prop = np.random.rand()
                p = quote_change_distribution(x, dist_params[0], dist_params[1], dist_params[2])
                if p/max_norm > acceptance_prop:
                    break
        yield x

def correlation_simulation_generator(dist_params, T):
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
        while True:
            x = 2 * np.random.rand() - 1.0
            acceptance_prop = np.random.rand()
            p = correlation_distribution(x, prop_param[0], prop_param[1])
            if p/max_norm > acceptance_prop:
                break
        yield x

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
    plt.xlabel("$T$")
    plt.ylabel("$f^0_T$")
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

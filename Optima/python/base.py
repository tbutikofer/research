import MySQLdb
import pandas as pd
import numpy as np
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

def cleanup_data(df):
    df = df.pivot(index='date', columns='isin', values='quote')
#    df.to_csv('quotes.csv')

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

def get_clean_data(marketcode, days, isin_list = None, today_str=date.today().strftime(DATE_FORMAT)):
    today = datetime.strptime(today_str, DATE_FORMAT)
    db_connection = get_db_connection()
    df = read_raw_data(db_connection, marketcode, days, isin_list, today)
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

def latex_table(table_array):
    print(r"\begin{{tabular}}{{{}}}".format(" | ".join(table_array[0])))
    row_template = " & ".join(["{}" for _ in table_array[0]]) + r"\\"
    column_names = table_array[1]
    print(row_template.format(*column_names))
    print(r"\hline")
    table_rows = table_array[2:]
    for table_row in table_rows:
        table_row_esc = [value.replace('&', '\\&') for value in table_row]
        print(row_template.format(*table_row_esc))
    print(r"\end{tabular}")

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
 
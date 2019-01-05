import python.base as base
import numpy as np
import itertools

def get_market_data(marketcode, series_length, until_date, isin_list):
    quote_array, date, isin, issuer = base.get_clean_data(marketcode, 3 * series_length, isin_list, until_date)
    quote = np.array(quote_array)
    quote_change = np.log(quote[1:] / quote[:-1])
    date = date[1:]
    quote = quote[1:]
    # remove days without trading (data_change = 0 for all securities)
    trading_days = ~np.all(quote_change == 0, axis=1)
    quote_change = quote_change[trading_days,:]
    quote_change = quote_change[-series_length:,:]
    quote = quote[trading_days]
    quote = quote[-series_length:]
    date = date[trading_days]
    date = date[-series_length:]
    # date: day , quote: quote at end of day, quote_change: change from day-1 to day
    return quote, quote_change, date, np.hstack((isin, issuer))

def trading(buy, cash, asset, quote):
    trading_fee = 30.0
    if buy:
        asset_change = np.floor((cash - trading_fee) / quote)
        cash_change = -np.round(asset_change * quote, 2)
    else:
        cash_change = np.round(asset * quote, 2)
        asset_change = -asset
    cash_change = cash_change - trading_fee
    return cash_change, asset_change

def find_optimal_trading_sequence():
    series_length = 20
    quote, quote_change, date, security = get_market_data('SWX',series_length+1,'15.10.2018',['CH0012221716'])
    init_cash = 10000.0
    init_asset = [0]
    max_gain = -100000.0
    for trade_sequence in itertools.product((False, True), repeat=series_length):
        cash = init_cash
        asset = [0]
        buy = True
        value_before = cash
        for i, trade in enumerate(trade_sequence):
            if trade:
                cash_change, asset_change = trading(buy, cash, asset[0], quote[i,0])
                cash = round(cash + cash_change, 2)
                asset[0] += asset_change
                buy = not buy
            value_after = cash + np.round(np.dot(asset, quote[i]), 2)
            value_before = value_after
        value_after = cash + np.round(np.dot(asset, quote[-1]), 2)
        gain = value_after/init_cash-1
        max_gain = max(max_gain, gain)
    #    print("Result: {:.2f}%".format(100*gain))
    print("Best Result: {:.2f}%".format(100*max_gain))
    # Best result 20 - 15.10.2018: 1.01%

data = base.get_securities('SWX')
ii = 42

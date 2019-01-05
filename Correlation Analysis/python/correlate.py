#import python.base as base
import python.base as base
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import shutil
from datetime import datetime, timedelta

def calc_momentum(corr, axis=None):
    momentum = [np.dot(corr[i],corr[i]) for i in range(len(corr))]
    if axis is None:
        momentum = np.sum(momentum)
    momentum = np.sqrt(momentum)/len(corr)
    return momentum

    # corr, _, _ = calc_correlation(marketcode, 28, date)
    # avg_momentum = np.sqrt(np.average( corr * corr))
    # print("{}: {:.4f}: {}".format(date, avg_momentum, corr.shape[0]))
    # return avg_momentum

def cluster_data(data, num_clusters):
    def input_fn():
        return tf.data.Dataset.from_tensors(
            tf.convert_to_tensor(data, dtype=tf.float32)
            ).repeat(count=1)
    #    return tf.train.limit_epochs(tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)

    shutil.rmtree('tmp_tf')
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False, model_dir='tmp_tf')    

    # train
    score_prev = 1.0
    while True:
        kmeans.train(input_fn)
    #    cluster_centers = kmeans.cluster_centers()
        score = kmeans.score(input_fn)
        if np.abs(1.0 - score/score_prev) < 0.000001:
            break
        score_prev = score

    cluster_indices = np.array(list(kmeans.predict_cluster_index(input_fn)))

    #reorder cluster indices
    cluster_norm = []
    for i in range(num_clusters):
        aa = data[cluster_indices==i,:][:,cluster_indices==i]
        cluster_norm.append(calc_momentum(aa))
    index_order = np.argsort(np.negative(cluster_norm))
    cluster_indices_sorted = np.zeros([len(cluster_indices)])
    for i in range(num_clusters):
        cluster_indices_sorted[cluster_indices==index_order[i]] = i
    return cluster_indices_sorted.astype(int)

def calc_correlation(marketcode, days, offset, today):
#    data_array, isin_desc = np.array(base.get_clean_data('SWX', 12, ['CH0012221716', 'CH0012255151']))
    data_array, _, isin_list, issuer_list = base.get_clean_data(marketcode, days=3 * days, today_str=today)
    data = np.array(data_array)
    data_change = np.log(data[1:] / data[:-1])
    # remove days without trading (data_change = 0 for all securities)
    trading_days = ~np.all(data_change == 0, axis=1)
    data_change = data_change[trading_days,:]
    data_change = data_change[-days:,:]

    if offset > 0:
        securities_traded = np.logical_and(
            ~np.all(data_change[offset:] == 0, axis=0),
            ~np.all(data_change[:-offset] == 0, axis=0)
        )
        data_change = data_change[:,securities_traded]
        isin_count = data_change.shape[1]
        corr = np.corrcoef(x=data_change[offset:], y=data_change[:-offset],rowvar=False)[:isin_count,-isin_count:]
    else:
        securities_traded = ~np.all(data_change == 0, axis=0)
        data_change = data_change[:,securities_traded]
    #    corr = np.cov(data_change,rowvar=False) / len(data_change)
        corr = np.corrcoef(data_change,rowvar=False)
        corr = corr - np.diag(np.ones(len(corr)))

    if len(corr.shape) == 0:
        corr = np.ones([1,1]) * corr
    #    np.savetxt("foo.csv", data_change, delimiter=",")
    isin_list = isin_list[securities_traded]
    issuer_list = issuer_list[securities_traded]
    trading_days = len(data_change)

    return corr, trading_days, isin_list, issuer_list

def get_first_day(last_day, interval_days):
    first_day = datetime.strptime(last_day, base.DATE_FORMAT)-timedelta(days=interval_days)
    return datetime.strftime(first_day, base.DATE_FORMAT)

def order_correlation_matrix(corr, num_clusters):
    cluster_idx = cluster_data(corr, num_clusters=num_clusters)

#    norm = calc_momentum(corr, axis=0)
    norm = np.average(corr, axis=0)
    corr_sorted_idx = np.lexsort([np.negative(norm), cluster_idx])
    corr_sorted = corr[corr_sorted_idx,:][:,corr_sorted_idx]
    return corr_sorted, cluster_idx, norm, corr_sorted_idx


def plot_correlation_matrix(corr, filename=None):
    plt.imshow(corr, cmap=plt.get_cmap('jet'))
    plt.clim(-1, 1)
    plt.colorbar()
    plt.axis('off')
    if filename is None:
        plt.show()
    else:
        plt.savefig('img/'+filename+".png", format='png')

def calculate_market_dynamics(marketcode, correlation_length, offset, date_series):
    momentum_series = []
    for date in date_series:
        date_str = date.strftime(base.DATE_FORMAT)
        corr, _, _, _ = calc_correlation(marketcode, correlation_length, offset, date_str)
        if corr is not None:
            momentum = calc_momentum(corr)
        else:
            momentum = 0.0
        momentum_series.append(momentum)
        print("{} {} {:.6f}".format(date.strftime(base.DATE_FORMAT), corr.shape[0], momentum))

    return momentum_series

def sample_quote_correlations(marketcode, days, offset, today, show_cluster, sample_size, filename=None):
    corr, trading_days, isin_list, issuer_list = calc_correlation(marketcode, days, offset, today)
    corr_sorted, cluster_idx, norm, corr_sorted_idx = order_correlation_matrix(corr, len(show_cluster))
    plot_correlation_matrix(corr_sorted, filename=filename)
    a = np.vstack((cluster_idx, isin_list, issuer_list, norm)).T[corr_sorted_idx]
    if filename is not None:
        np.savetxt('csv/'+filename+".csv", [['cluster', 'isin', 'issuer', 'norm']] + [column.tolist() for column in a], fmt='%s', delimiter=",")

    isin_sample_list = []
    for i, _ in enumerate(show_cluster):
        isin_sample_list.append(a[a[:,0]==str(i)][:sample_size][:,1])
    isin_sample_list = np.array(isin_sample_list)
    values, dates, isin_list, _ = base.get_clean_data(marketcode, days, np.hstack(isin_sample_list), today)
    values = np.array(values)
    data_change = np.log(values[1:] / values[:-1])
    dates = dates[1:]
    # remove days without trading (data_change = 0 for all securities)
    trading_days = ~np.all(data_change == 0, axis=1)
    data_change = data_change[trading_days,:]
    dates = dates[trading_days]

    sample_data = []
    for i, _ in enumerate(show_cluster):
        cluster_data = {'id':str(i), 'data':[]}
        cluster_data['norm'] = np.average(a[a[:,0] == str(i),3].astype(float))
        for isin in isin_sample_list[i]:
            data_set = {
                'id':isin,
                'norm':a[a[:,1] == isin][0,3],
                'issuer':a[a[:,1] == isin][0,2],
                'data':np.squeeze(data_change[:,isin_list == isin])}
            cluster_data['data'].append(data_set)
        sample_data.append(cluster_data)

    legend = np.array(["{} ({:.4})".format(cluster['id'], cluster['norm']) for cluster in sample_data])
    legend[np.logical_not(show_cluster)] = ''
#    plot_timeseries(dates, sample_data, legend, filename=filename+"_sample")

    return sample_data, dates, len(a)

def find_base_securities(marketcode, events):
    securities = base.get_securities(marketcode)
    days = 28
    offset = 0
    num_clusters = 4
    isin_cluster_events = []
    for event in events:
        isin_cluster = np.empty(len(securities)) * np.nan
        corr, trading_days, isin_list, _ = calc_correlation(marketcode, days, offset, event['date'])
        cluster_idx = cluster_data(corr, num_clusters=num_clusters)
        for i in range(num_clusters):
            isin_cluster[np.in1d(securities[:,0], isin_list[cluster_idx == i])] = i
        isin_cluster_events.append(isin_cluster)
    isin_cluster_events = np.transpose(np.array(isin_cluster_events))
    all_clusters_defined = np.logical_not(np.any(np.isnan(isin_cluster_events), axis=1))
    securities = securities[all_clusters_defined]
    isin_cluster_events = isin_cluster_events[all_clusters_defined]
    print(securities[np.all(isin_cluster_events == 0, axis=1)])

def animate_market_dynamics(marketcode, start_date_str, end_date_str, correlation_length, offset):
    start_date = datetime.strptime(start_date_str, base.DATE_FORMAT)
    end_date = datetime.strptime(end_date_str, base.DATE_FORMAT)
    date_range = (start_date + timedelta(x) for x in range((end_date-start_date).days+1))
#    date_range = (x for x in range(start_date, end_date)
    #    date_range = (start_date - timedelta(x-1) for x in range(days, 0, -1))

    fig, ax = plt.subplots()
    im = plt.imshow(np.random.randn(150,150), cmap=plt.get_cmap('jet'), animated=True)
    plt.axis('off')
    plt.clim(-1, 1)

    def init():
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
        ax.set_ylim(ax.get_ylim()[::-1])         
        return im,

    def update(date):
        date_str = date.strftime(base.DATE_FORMAT)
        print(date_str)
        corr, _, _, _ = calc_correlation(marketcode, correlation_length, offset, date_str)
        norm = np.average(corr, axis=0)
        corr_sorted_idx = np.lexsort([np.negative(norm)])
#        cluster_idx = cluster_data(corr, num_clusters=4)
#        corr_sorted_idx = np.lexsort([np.negative(norm), cluster_idx])
        corr_sorted = corr[corr_sorted_idx,:][:,corr_sorted_idx]
        plt.title("{}".format(date_str))
        im.set_array(corr_sorted)

        return im,

    anim = animation.FuncAnimation(fig, update, frames=date_range, init_func=init, repeat=False, blit=True, save_count=(end_date-start_date).days)
    filename = "market_correlation_{}_{}-{}".format(marketcode, base.date_stamp(start_date_str),base.date_stamp(end_date_str))
    anim.save("img/{}.mp4".format(filename), fps=30, extra_args=['-vcodec', 'libx264'])

def calc_market_correlations(marketcode, start_date_str, end_date_str, step_days, correlation_lengths, offset):
    date_series = base.create_date_series(start_date_str, end_date_str, step_days)
    header = ['date']
    data_series = np.array([[date.strftime(base.DATE_FORMAT) for date in date_series]])
    for correlation_length in correlation_lengths:
        header.append(str(correlation_length))
        momentum_series = calculate_market_dynamics(marketcode, correlation_length, offset, date_series)
        data_series = np.append(data_series, [momentum_series], axis=0)
    #    data_series.append({'id':correlation_length, 'data':[{'data':momentum_series}]})
    data_csv = np.append(np.array([header]),np.transpose(data_series),axis=0)   
    filename = "market_correlation_{}_{}-{}".format(marketcode, base.date_stamp(start_date_str),base.date_stamp(end_date_str))
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')
    return filename

def plot_time_series(filename, legends, events=None):
    data_csv = np.loadtxt('csv/'+filename+'.csv', dtype=np.unicode, delimiter=',', skiprows=1)
    date_series = [datetime.strptime(date_str, base.DATE_FORMAT) for date_str in data_csv[:,0]]
    data = data_csv[:,1:].astype(np.float)
    color_list = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
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
            label=legend_txt)

    ylim = ax.get_ylim()
    if events is not None:
        for event in events:
            date = datetime.strptime(event['date'], base.DATE_FORMAT)
            plt.axvline(x=date, linestyle=':', color='black')
            plt.text(date,ylim[1]-(ylim[1]-ylim[0])/50.0,event['label'],va='top')

    legend = ax.legend()
    if filename is None:
        plt.show()
    else:
        plt.savefig('img/'+filename+".png", format='png')
    legend = ax.legend().remove()
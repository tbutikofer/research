import python.base as base
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import shutil
import tensorflow as tf

def pearson_correlation(time_series, offset=0):
#    np.savetxt('csv/pearsoncorrelaton.csv', time_series, fmt='%s', delimiter=',')

    if time_series.shape[1] > 1:
        if offset > 0:
            series_count = time_series.shape[1]
            corr = np.corrcoef(x=time_series[offset:], y=time_series[:-offset],rowvar=False)[:series_count,-series_count:]
        else:
            corr = np.corrcoef(time_series,rowvar=False)
    else:
        corr = None
    return corr

def calculate_stress(corr, axis=None):
    corr = corr - np.diag(np.ones(len(corr)))
    stress = [np.dot(corr[i],corr[i]) for i in range(len(corr))]
    if axis is None:
        stress = np.sum(stress)
    stress = np.sqrt(stress)/len(corr)
    return stress

def stress_series(change_generator, offset, filename):
    stress_series = []
    date_series = []
    for data_change, date in change_generator:        
        # remove securites which haven't been traded during this time period
        securities_traded = np.logical_and(
            ~np.all(data_change == 0, axis=0),
            ~np.any(np.isnan(data_change), axis=0)
        )
        data_change = data_change[:,securities_traded]

        corr = pearson_correlation(data_change, offset)
        if corr is not None:
            stress = calculate_stress(corr)
        else:
            stress = 0.0
        stress_series.append(stress)
        date_series.append(date)
        security_count = corr.shape[0] if corr is not None else 0
        # print("{} {} {:.6f}".format(date.strftime(base.DATE_FORMAT), security_count, stress))

    header = ['date', 'stress']
    data_series = np.vstack(([date.strftime(base.DATE_FORMAT) for date in date_series], stress_series)).T
    data_csv = np.append(np.array([header]),data_series,axis=0)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')

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
        cluster_norm.append(calculate_stress(aa))
    index_order = np.argsort(np.negative(cluster_norm))
    cluster_indices_sorted = np.zeros([len(cluster_indices)])
    for i in range(num_clusters):
        cluster_indices_sorted[cluster_indices==index_order[i]] = i
    return cluster_indices_sorted.astype(int)

def market_stress_matrix(marketcode, date, T, filename):
    num_clusters = 4
    change_generator, security_list = base.observed_changes(marketcode, (date,date), T=T)
    change_data, date = next(change_generator)
    corr = pearson_correlation(change_data)
    order_metric = np.average(corr, axis=0)
    cluster_idx = cluster_data(corr, num_clusters=num_clusters)
    data_sorted_idx = np.lexsort([np.negative(order_metric), cluster_idx])
    corr = corr[data_sorted_idx,:][:,data_sorted_idx]
    security_list = security_list[data_sorted_idx]

    header = ['isin'] + [security[0] for security in security_list]
    data_csv = np.array([header])
    for security_idx, security in enumerate(security_list):
        data_csv = np.append(data_csv, np.concatenate(([[security[0]]], [corr[security_idx]]), axis=1), axis=0)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')

def find_base_securities(marketcode, event_filename, date_range, T, filename):
    event_data = np.loadtxt('csv/'+event_filename+'.csv', dtype=np.unicode, delimiter=',')[1:,:]
    date_from = datetime.strptime(date_range[0], base.DATE_FORMAT)
    date_to = datetime.strptime(date_range[1], base.DATE_FORMAT)
    event_date_list = []
    for event in event_data:
        date = datetime.strptime(event[1], base.DATE_FORMAT)
        if date_from <= date <= date_to:
            event_date_list.append(date.date())

    offset = 0
    num_clusters = 4
    event_security_cluster = []

    change_generator, security_list = base.observed_changes(marketcode, date_range, always=True, T=T)

    event_idx = 0
    event_date = event_date_list[event_idx]
    last_data_change = np.array([])
    for data_change, date in change_generator:
        if date > event_date:
            traded_securities_ids = ~np.all(last_data_change == 0, axis=0)
            effective_change = last_data_change[:,traded_securities_ids]
            traded_securities = security_list[traded_securities_ids]
            corr = pearson_correlation(effective_change, offset)
            cluster_idx = cluster_data(corr, num_clusters=num_clusters)

            security_clusters = np.empty(len(security_list)) * np.nan
            for i in range(num_clusters):
                security_clusters[np.in1d(security_list[:,0], traded_securities[cluster_idx == i,0])] = i
            event_security_cluster.append(security_clusters)
            event_idx = event_idx + 1
            if event_idx >= len(event_date_list):
                break
            event_date = event_date_list[event_idx]
        last_data_change = data_change
    event_security_cluster = np.array(event_security_cluster)

    header = ['isin','issuer']
    data_csv = np.array([header])
    for row in security_list[np.all(event_security_cluster == 0, axis=0)]:
        data_csv = np.append(data_csv, [row], axis=0)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')

def calculate_histogram(change_generator, change_range, filename, sample_max = None):
    data_change = []
    for data, _ in change_generator:
        data_change.append(data)
        if sample_max and len(data_change) >= sample_max:
            break
    data_change = np.array(data_change).flatten()
    hist, bins = np.histogram(data_change,bins=100, range = change_range, density=True)
    bins_avg = (bins[1:]+bins[:-1])/2

    np.savetxt('csv/'+filename+'.csv', np.vstack((bins_avg,hist)).T, fmt='%s', delimiter=',')
    return len(data_change)

def baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, change_generator, filename):
    """
    Calculates baseline stress from synthetic quote changes.
    Determines base.distribution_correlation for synthetic correlation matrix.
    """
    def prob_dist_T1(x, a, b):
        return np.power(x, a) * b

    def prob_dist_T2(x, a, b):
        return a * x + b


    data_record = ['trading days', 'average stress', 'std stress']
    data_record = np.concatenate((data_record, ["fit param {}".format(i+1) for i in range(2)]))
    data = np.array([data_record])
    for trading_days in trading_days_list:
        quote_change = base.quote_change_simulation_generator(change_generator, securities, T=trading_days)
        corr_elements = []
        stress_series = []
        for _ in range(simulation_runs):
            data_change, _ = next(quote_change)
            corr = pearson_correlation(data_change, offset=0)
            for i in range(securities-1):
                for j in range(i+1, securities):
                    corr_elements.append(corr[i,j])
            stress = calculate_stress(corr)
            stress_series.append(stress)
        data_record = [trading_days, np.average(stress_series), np.std(stress_series)]
        hist, bins = np.histogram(corr_elements, bins=100, density=True)
        bins_avg = (bins[1:]+bins[:-1])/2
        popt, pcov = curve_fit(base.correlation_distribution, bins_avg, hist, p0=(1.9, 20.0))
        data_record = np.concatenate((data_record, popt))
        data = np.append(data, [data_record], axis=0)

    x = data[1:,0].astype(np.float)
    p1 = data[1:,3].astype(np.float)
    p2 = data[1:,4].astype(np.float)

    popt1, pcov1 = curve_fit(prob_dist_T1, x, p1, p0=(0.3, 0.5))
    popt2, pcov2 = curve_fit(prob_dist_T2, x, p2, p0=(0.4, -1.5))

    np.savetxt('csv/'+filename+'.csv', data, fmt='%s', delimiter=',')
    return (popt1, np.sqrt(np.diag(pcov1))), (popt2, np.sqrt(np.diag(pcov2)))

def baseline_stress_correlation_simulation(securities, simulations, T):
    """ Calculates baseline stress from synthetic correlation matrix """
    correlation_generator = base.correlation_simulation_generator((0.52587211, 0.34470357, 0.40524949 , -1.46300512), securities, T)
    momentum_series = []
    for _ in range(simulations):
        corr = next(correlation_generator)
        stress = calculate_stress(corr)
        momentum_series.append(stress)
    return np.average(momentum_series)

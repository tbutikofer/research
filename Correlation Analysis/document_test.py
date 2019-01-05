# correlation_interval = 14
# corr, trading_days, isin_list, issuer_list = correlate.calc_correlation('SWX', correlation_interval, 0, '07.01.2018')
# num_clusters = 2
# cluster_idx, norm, sorted_idx = correlate.plot_correlation_matrix(corr, num_clusters)

# a = np.vstack((cluster_idx, isin_list, issuer_list, norm)).T[sorted_idx]
# #np.savetxt("foo.csv", a, fmt=['%s','%s','%s','%s'], delimiter=",")

# isin_list = []
# for i in range(num_clusters):
#     isin_list.extend(a[a[:,0]==str(i)][:5][:,1])
# data_array, isin_list, _ = base.get_clean_data('SWX', correlation_interval, isin_list=isin_list, today_str='31.12.2017')
# data = np.array(data_array)
# data_change = np.log(data[1:] / data[:-1])
# time_series = {'date':[i-1 for i in range(data.shape[0]-1, 0, -1)]}
# series_legend = [{'legend':isin, 'data':isin} for isin in isin_list]
# value_series = {}
# for isin_idx, isin in enumerate(isin_list):
#     value_series[isin] = data_change[:,isin_idx]
# correlate.plot_timeseries(time_series, value_series, series_legend)

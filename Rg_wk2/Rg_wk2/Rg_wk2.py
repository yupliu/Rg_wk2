import graphlab
sales = graphlab.SFrame('D:\\ML_Learning\\UW_Regression\\Week2\\kc_house_data.gl\\')

sales['bedrooms_squared'] = sales['bedrooms']*sales['bedrooms']
sales['bed_bath_rooms'] = sales['bedrooms']*sales['bathrooms']
sales['log_sqft_living'] = log(sales['sqft_living'])
sales['lat_plus_long'] = sales['lat'] + sales['long']

train_data,test_data = sales.random_split(.8,seed=0)

print test_data['bedrooms_squared'].mean()
print test_data['bed_bath_rooms'].mean()
print test_data['log_sqft_living'].mean()
print test_data['lat_plus_long'].mean()

#test_data['bedrooms_squared_1'] = test_data['bedrooms'].apply(lambda x: x**2)
#print test_data['bedrooms_squared_1'].mean()

m1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
m2_features = m1_features + ['bed_bath_rooms']
m3_features = m2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


m1_model = graphlab.linear_regression.create(train_data, target = 'price', features = m1_features, 
                                                  validation_set = None)
m2_model = graphlab.linear_regression.create(train_data, target = 'price', features = m2_features, 
                                                  validation_set = None)
m3_model = graphlab.linear_regression.create(train_data, target = 'price', features = m3_features, 
                                                  validation_set = None)
print m1_model.coefficients
print m2_model.coefficients



def get_rss(model,data,target):
    rss_tmp = model.predict(data)
    #print rss_tmp
    rss_tmp = rss_tmp - target
    #print rss_tmp
    rss_tmp = rss_tmp * rss_tmp
    #print rss_tmp
    return rss_tmp.sum()    

rss_m1_test = get_rss(m1_model, test_data, test_data['price'])
print rss_m1_test

rss_m2_test = get_rss(m2_model, test_data, test_data['price'])
print rss_m2_test

rss_m3_test = get_rss(m3_model, test_data, test_data['price'])
print rss_m3_test

rss_m1_train = get_rss(m1_model, train_data, train_data['price'])
print rss_m1_train

rss_m2_train = get_rss(m2_model, train_data, train_data['price'])
print rss_m2_train

rss_m3_train = get_rss(m3_model, train_data, train_data['price'])
print rss_m3_train
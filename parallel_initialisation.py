#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:21:24 2018

@author: krupa
"""
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from initial_encoding_routes import greedy_nearest_neighbour_heauristic, nearest_neighbour_heauristic, random_neighbour_heauristic
#%%

#Change - Andre'


def calculate_early_delay_penalty(route_routeList_list, arr_distance_matrix, df_customers):
    early_time = 0
    delay_time = 0

    for route_list in route_routeList_list:  # loop over routes
        #print("new route")  # ADDED MS
        cust = route_list[1]
        total_dist = arr_distance_matrix[0, cust]
        readyTime = df_customers.loc[cust, 'readyTime']
        #Change - Andre'
        dueTime = df_customers.loc[cust, 'dueTime']
        if readyTime - total_dist > 0:
            start_time = readyTime - total_dist
        else:
            start_time = 0
        #print(start_time)  # ADDED MS
        curr_time = start_time + total_dist
        #print(curr_time)  # ADDED MS
        if curr_time < readyTime:
            early_time = early_time + (readyTime - curr_time)
        if curr_time > dueTime:
            delay_time = delay_time + (curr_time - dueTime)
        curr_time = curr_time + df_customers.loc[cust, 'serviceTime']

        for i in range(1, len(route_list) - 2):
            cust1 = route_list[i]
            cust2 = route_list[i + 1]
            dist = arr_distance_matrix[cust1, cust2]
            readyTime = df_customers.loc[cust2, 'readyTime']
            dueTime = df_customers.loc[cust2, 'dueTime']
            curr_time = curr_time + dist
            #print(curr_time)  # ADDED MS
            if curr_time < readyTime:
                early_time = early_time + (readyTime - curr_time)
            if curr_time > dueTime:
                delay_time = delay_time + (curr_time - dueTime)
            curr_time = curr_time + df_customers.loc[cust2, 'serviceTime']
    return (0.5 * early_time) + (2.5 * delay_time)


def initial_population(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot, num_cores, num_greedy_particles, M, num_random_particles, num_nnh_particles, e, p):


    #get individual result list of tuples
    greedy_result_list = Parallel(n_jobs = num_cores)(delayed(greedy_nearest_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_greedy_particles))
    nnh_result_list = Parallel(n_jobs = num_cores)(delayed(nearest_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_nnh_particles))
    random_result_list = Parallel(n_jobs = num_cores)(delayed(random_neighbour_heauristic)(df_customers,arr_customers, arr_distance_matrix, total_time, total_capacity, num_customers_depot)for i in range(num_random_particles))
    #group all tuples together in one list
    result_list = greedy_result_list + nnh_result_list
    result_list += random_result_list
    
    unzipped = zip(*result_list) #groups the same return for each particle and returns a list of tuples each of lenth M
    PSO_resultlist = list(unzipped)  # route_routeList_list, route_distance_list, particle_position, particle_velocity, total_distance, route

    arr_population_routeList = np.empty(M, object)
    arr_population_routeList[:] = PSO_resultlist[0][:]
    arr_population_distanceList = np.empty(M, object)
    arr_population_distanceList[:] = PSO_resultlist[1][:]
    arr_population_particle_position = np.empty(M, object)
    arr_population_particle_position[:] = PSO_resultlist[2][:]
    arr_population_particle_velocity = np.empty(M, object)
    arr_population_particle_velocity[:] = PSO_resultlist[3][:]    
    df_results = pd.DataFrame(np.zeros((M,3)))
    df_results.columns = ['num_vehicles', 'distance', 'fitness']
    df_results['num_vehicles'] = PSO_resultlist[5][:]
    df_results['distance'] = PSO_resultlist[4][:]


    #GA FITNESS
    alpha = 50
    beta = 0.14
    
    
    # Change - Andre'
    List_Penalty = []
    for i in range(len(PSO_resultlist[0])):
        Total_Penalty = calculate_early_delay_penalty(PSO_resultlist[0][i], arr_distance_matrix, df_customers)
        List_Penalty.append(Total_Penalty)

    arr_distance = np.array(PSO_resultlist[4]).copy()
    arr_distance = beta*arr_distance
    arr_vehicles = alpha*np.array(PSO_resultlist[5])
    #Change - Andre'
    #df_results['fitness'] = arr_distance+arr_vehicles
    df_results['fitness'] = arr_distance + arr_vehicles + np.array(List_Penalty)
    #print("d:",arr_distance,"v:",arr_vehicles,"p:",List_Penalty)  # ADDED MS
    
    return df_results, arr_population_routeList, arr_population_distanceList, arr_population_particle_position, arr_population_particle_velocity
    


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Created on Wed Dec  7 16:34:57 2016
    
    @author: kasperipalkama
    """

import numpy as np
import networkx as nx
import si_animator
import random
import progressbar
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


def initialize_nodes(net,first_infected,task_number = 0):
    infection_times = np.zeros(net.number_of_nodes())
    for index,node in enumerate(net.nodes()):
        #        print index, node
        if int(node) == first_infected:
            nodeIdx = np.argwhere(event_data['Source'] == int(node))[0][0]
            if task_number == 6:
                infection_times[int(node)] = int(node)
            else:
                infection_times[int(node)] = event_data['StartTime'][nodeIdx]
        else:
            infection_times[int(node)] = np.Inf
    return infection_times

def refresh_infections(infection_times,source,destination,startTime,endTime,probability_for_infection,infection_spreaders = 0,task_number = 0,immunized_nodes = []):
    if source not in immunized_nodes and destination not in immunized_nodes:
        if np.isinf(infection_times[source]) == False and startTime >= infection_times[source]:
            p = random.random()
            if p >= (1.0-probability_for_infection):
                if np.isinf(infection_times[destination]) or infection_times[destination] >= endTime:
                    infection_times[destination] = endTime
                    if task_number == 6:
                        infection_spreaders[destination] = source
    if task_number == 6:
        return infection_spreaders
    else:
        return infection_times

def fraction_infected(infection_times,simulationStartTime, simulationEndTime,binWidth):
    fractionInfected = []
    simulationTimeList = []
    simulationTime = simulationStartTime
    while simulationTime <= simulationEndTime:
        infected = infection_times <= simulationTime
        fractionInfected.append(sum(infected)/float(len(infection_times)))
        simulationTimeList.append(simulationTime)
        simulationTime += binWidth
    return np.array(fractionInfected),np.array(simulationTimeList)


def infected_visualize(y,x,label,xlabel,ylabel,title,task_number):
    if task_number==2:
        plt.figure(1)
    if task_number == 3:
        plt.figure()
    plt.plot(x,y,label = 'p ='+label)
    plt.legend(loc = 'best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def get_median_infection_time(infection_time_matrix, minValueCount):
    median_infection_time = np.empty(len(infection_time_matrix[:,0]))
    for index,_ in enumerate(infection_time_matrix[:,0]):
        node_times = infection_time_matrix[index,:]
        if len(node_times[node_times != np.inf]) > minValueCount:
            median_infection_time[index] = np.median(node_times)
    return median_infection_time

def get_network_measures(network):
    degree_dict = nx.degree(network)
    clustering_coef_dict = nx.clustering(network)
    betweenness_dict = nx.betweenness_centrality(network)
    strength_dict = nx.degree(network, weight = "weight")
    closeness_dict = nx.closeness_centrality(network)
    kshell_dict = nx.core_number(network)
    
    nodeCount = network.number_of_nodes()
    kshell = np.empty(nodeCount)
    clustering_coef = np.empty(nodeCount)
    degree = np.empty(nodeCount)
    strength = np.empty(nodeCount)
    betweenness = np.empty(nodeCount)
    closeness = np.empty(nodeCount)
    
    for node in network.nodes():
        kshell[int(node)] = kshell_dict[node]
        clustering_coef[int(node)] = clustering_coef_dict[node]
        degree[int(node)] = degree_dict[node]
        strength[int(node)] = strength_dict[node]
        betweenness[int(node)] =        [node]
        closeness[int(node)] = closeness_dict[node]
    return [kshell, clustering_coef, degree, strength, betweenness, closeness]

def visualize(y,x,xlabel,ylabel,mark,task_number,title = '',label = ''):
    if task_number == 5:
        fig = plt.figure(1)
    if task_number == 4 or task_number == 6 :
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,y,mark,label = label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if task_number == 5:
        ax.legend(loc='best',fontsize=10,fancybox=True).get_frame().set_alpha(0.5)


def get_immunized_nodes(network,immunization_strategies,nodeCount,measures):
    immunized_nodes_matrix = []
    for immunization_strategy in immunization_strategies:
        immunized_nodes = []
        if immunization_strategy == 'rand_node':
            immunized_nodes = random.sample(range(network.number_of_nodes()), nodeCount)
        if immunization_strategy == 'rand_node_neighbor':
            rand_nodes = random.sample(network.nodes(),nodeCount)
            for rand_node in rand_nodes:
                nodeToImmunize = int(random.sample(nx.neighbors(network,rand_node),1)[0])
                immunized_nodes.append(nodeToImmunize)
        if immunization_strategy == 'kshell':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[0])
                measures[0][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        if immunization_strategy == 'c':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[1])
                measures[1][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        if immunization_strategy == 'k':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[2])
                measures[2][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        if immunization_strategy == 's':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[3])
                measures[3][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        if immunization_strategy == 'bc':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[4])
                measures[4][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        if immunization_strategy == 'cc':
            for index in range(nodeCount):
                maxIndex = np.argmax(measures[5])
                measures[5][maxIndex] = 0
                immunized_nodes.append(maxIndex)
        immunized_nodes_matrix.append(immunized_nodes)
    return immunized_nodes_matrix

def get_seed_nodes(network,immunized_nodes,numer_of_seed_nodes):
    seedNodes = []
    for index in range(numer_of_seed_nodes):
        random_node = int(random.sample(network.nodes(),1)[0])
        while any(random_node in sublist for sublist in immunized_nodes) and random_node not in seedNodes:
            random_node = int(random.sample(network.nodes(),1)[0])
        seedNodes.append(random_node)
    return seedNodes

def get_link_infecting_frequency(network,infection_spreaders_matrix):
    fij = np.zeros(network.number_of_edges())
    infection_spreaders_matrix = np.array(infection_spreaders_matrix)
    n = float(len(infection_spreaders_matrix[:,0]))
    for index,edge in enumerate(network.edges()):
        startNode = int(edge[0])
        endNode = int(edge[1])
        for nodeS in range(len(infection_spreaders_matrix[0,:])):
            if int(nodeS) == startNode:
                infections = list(infection_spreaders_matrix[:,nodeS])
                linkCount1 = infections.count(endNode)
            if int(nodeS) == endNode:
                infections = list(infection_spreaders_matrix[:,nodeS])
                linkCount2 = infections.count(startNode)
        fij[index] = linkCount1/n
        fij[index] += linkCount2/n
    
    return list(fij)

def visualize_links(network,linkwidths = None):
    xycoords = np.loadtxt('US_airport_id_info.csv',skiprows=1,delimiter=',',usecols=(6,7))
    nodes = network.nodes()
    nodeIdxs = []
    for node in network.nodes():
        nodeIdxs.append(int(node))
    nodesInOrder = [x for (y,x) in sorted(zip(nodeIdxs,nodes))]
    xycoords_dict = dict(zip(nodesInOrder,xycoords))
    for key in xycoords_dict.keys():
        xycoords_dict[key] = list(xycoords_dict[key])
    si_animator.plot_network_usa(net=network,xycoords=xycoords_dict,linewidths=linkwidths,edges=network.edges())

def maximum_spanning_tree(network):
    G = network.copy()
    for (i,j,data) in G.edges(data = True):
        data['weight'] = -1 * data['weight']
    return nx.minimum_spanning_tree(G)

def get_link_weigth(network):
    weights = []
    for (i,j,data) in network.edges(data = True):
        weights.append(data['weight'])
    return weights

def get_link_overlap(network):
    overlaps = []
    for (i,j,data) in network.edges(data = True):
        n_ij = len(sorted(nx.common_neighbors(network,i,j)))
        k_i = nx.degree(network,i)
        k_j = nx.degree(network,j)
        if k_i == 1 and k_j == 1:
            overlaps.append(0)
        else:
            overlaps.append(n_ij/float(k_i - 1 + k_j - 1 + n_ij))
    return overlaps

def get_edge_betweennes_centrality_as_list(network):
    edge_betweennes_dict = nx.edge_betweenness_centrality(network)
    edge_betweennes = []
    for edge in network.edges():
        edge_betweennes.append(edge_betweennes_dict[edge])
    return edge_betweennes

def get_bonus_measure1(network):
    bonus = []
    edge_betweennes = get_edge_betweennes_centrality_as_list(network)
    weights = get_link_weigth(network)
    for index,edge in enumerate(network.edges()):
        bonus.append(weights[index]*edge_betweennes[index])
    return bonus

def get_bonus_measure2(network):
    bonus = []
    edge_betweennes = get_edge_betweennes_centrality_as_list(network)
    weights = get_link_weigth(network)
    for index,edge in enumerate(network.edges()):
        smaller_node_degree = max(1/float(nx.degree(network,edge[0])),1/float(nx.degree(network,edge[1])))
        bonus.append(weights[index]*smaller_node_degree*edge_betweennes[index])
    return bonus


def Task1(event_data,network):
    seedNode = 0
    prob = 1.0
    infection_times = initialize_nodes(network,first_infected=seedNode)
    for source,destination,startTime,endTime,duration in zip(event_data['Source'],event_data['Destination'],event_data['StartTime'],event_data['EndTime'],event_data['Duration']):
        infection_times = refresh_infections(infection_times,source,destination,startTime,endTime,probability_for_infection = prob)
    print infection_times[41]

def Task2and3(event_data,network,task_number):
    if task_number == 2:
        p = [0.01, 0.05, 0.1, 0.5, 1.0]
        titles = ['','','','','']
        seedNode = [0,0,0,0,0,0]
    else:
        p = [0.1,0.1,0.1,0.1,0.1]
        titles = ['ABE','ATL','ACV','HSV','DBQ']
        seedNode = [0,4,41,100,200]
    wb = progressbar.ProgressBar(maxval=len(p))
    wb.start()
    binWidth = 1000
    simulationStartTime = event_data['StartTime'][0]
    simulationEndTime = np.max(event_data['EndTime'])
    for index,prob in enumerate(p):
        n = 10
        fractionInfectedMatrix = np.empty([(simulationEndTime-simulationStartTime)/binWidth + 1,n])
        for i in range(n):
            infection_times = initialize_nodes(network,first_infected=seedNode[index])
            for source,destination,startTime,endTime,duration in zip(event_data['Source'],event_data['Destination'],event_data['StartTime'],event_data['EndTime'],event_data['Duration']):
                infection_times = refresh_infections(infection_times,source,destination,startTime,endTime,probability_for_infection = prob)
            fractionInfected,time = fraction_infected(infection_times,simulationStartTime, simulationEndTime,binWidth)
            fractionInfectedMatrix[:,i] = fractionInfected
        fractionInfectedAverage = np.mean(fractionInfectedMatrix,axis = 1)
        infected_visualize(fractionInfectedAverage,time,str(prob),'time','fraction infected',titles[index],task_number)
        wb.update(index)
    wb.finish()

def Task4(event_data,network):
    prob = 0.5
    n = 50
    infection_times_matrix = np.empty([network.number_of_nodes(),n])
    wb = progressbar.ProgressBar(maxval=n-1)
    for index in range(n):
        wb.start()
        seedNode = random.randint(0,network.number_of_nodes()-1)
        infection_times = initialize_nodes(network,first_infected=seedNode)
        for source,destination,startTime,endTime,duration in zip(event_data['Source'],event_data['Destination'],event_data['StartTime'],event_data['EndTime'],event_data['Duration']):
            infection_times = refresh_infections(infection_times,source,destination,startTime,endTime,probability_for_infection = prob)
        infection_times_matrix[:,index] = infection_times
        wb.update(index)
    wb.finish()
    median_infection_time = get_median_infection_time(infection_times_matrix,25)
    measures = get_network_measures(network)
    xlabels = ['k-shell', 'clustering coefficient', 'degree', 'strength', 'betweennes centrality','closeness centrality']
    for measure,xlabel in zip(measures,xlabels):
        visualize(y=median_infection_time,x=measure,xlabel=xlabel,ylabel='median infection time',mark='o',task_number=4)
        corr, _ = spearmanr(measure, median_infection_time)
        print 'Spearman correlation coefficient of',xlabel,':',corr

def Task5(event_data, network):
    meausures = get_network_measures(network)
    binWidth = 1000
    prob = 0.5
    n = 20
    wb = progressbar.ProgressBar(maxval=8)
    immunization_strategies = ['rand_node_neighbor','rand_node','kshell','c','k','s','bc','cc']
    immunized_nodes_matrix = get_immunized_nodes(network=network,immunization_strategies=immunization_strategies,nodeCount=10,measures=meausures)
    seedNodes = get_seed_nodes(network=network,immunized_nodes=immunized_nodes_matrix,numer_of_seed_nodes=n)
    for i,immunized_nodes in enumerate(immunized_nodes_matrix):
        #        print immunized_nodes
        wb.start()
        simulationStartTime = event_data['StartTime'][0]
        simulationEndTime = max(event_data['EndTime'])
        fractionInfectedMatrix = np.zeros([np.round((simulationEndTime-simulationStartTime)/binWidth)+1,n])
        for index,seedNode in enumerate(seedNodes):
            infection_times = initialize_nodes(network,first_infected=seedNode)
            for source,destination,startTime,endTime,duration in zip(event_data['Source'],event_data['Destination'],event_data['StartTime'],event_data['EndTime'],event_data['Duration']):
                infection_times = refresh_infections(infection_times,source=source,destination=destination,startTime=startTime, endTime=endTime,probability_for_infection = prob,immunized_nodes=immunized_nodes)
            fractionInfected, time = fraction_infected(infection_times,simulationStartTime,simulationEndTime,binWidth = binWidth)
            fractionInfectedMatrix[:,index] = fractionInfected
        visualize(np.mean(fractionInfectedMatrix,axis = 1),time, xlabel='time',ylabel = 'fraction infected',mark = '-',label=immunization_strategies[i],task_number = 5)
        wb.update(i+1)
    wb.finish()

def Task6(event_data, network):
    prob = 0.5
    n = 20
    infection_spreaders_matrix = []
    wb = progressbar.ProgressBar(maxval=n)
    wb.start()
    for index in range(n):
        seedNode = int(random.sample(network.nodes(),1)[0])
        infection_times = initialize_nodes(net=network,first_infected=seedNode)
        infection_spreaders = initialize_nodes(net=network,first_infected=seedNode,task_number=6)
        for source,destination,startTime,endTime,duration in zip(event_data['Source'],event_data['Destination'],event_data['StartTime'],event_data['EndTime'],event_data['Duration']):
            infection_spreaders = refresh_infections(infection_times=infection_times,source=source,destination=destination,startTime=startTime, endTime=endTime,probability_for_infection = prob,task_number=6,infection_spreaders=infection_spreaders)
            infection_times = refresh_infections(infection_times=infection_times,source=source,destination=destination,startTime=startTime, endTime=endTime,probability_for_infection = prob)
        infection_spreaders_matrix.append(infection_spreaders)
        wb.update(index+1)
    wb.finish()
    fij = get_link_infecting_frequency(network,infection_spreaders_matrix)
    max_span_tree = maximum_spanning_tree(network)
    visualize_links(max_span_tree)
    visualize_links(network=network,linkwidths=fij)
    
    weights = get_link_weigth(network)
    overlaps = get_link_overlap(network)
    edge_betweennes = get_edge_betweennes_centrality_as_list(network)
    links_measures = [weights, overlaps, edge_betweennes]
    xlabels = ['weigth', 'link neighborhood overlap', 'unweighted edge betweennes centrality']
    for link_measure,xlabel in zip(links_measures,xlabels):
        visualize(y=fij,x=link_measure,xlabel=xlabel,ylabel='fij',mark='o',task_number=6)
        corr,_ = spearmanr(fij,link_measure)
        print 'Spearman corr(fij,',xlabel,') = ',corr
    bonus1 = get_bonus_measure1(network)
    bonus2 = get_bonus_measure2(network)
    corrbonus1,_ = spearmanr(fij,bonus1)
    corrbonus2,_ = spearmanr(fij,bonus2)
    print 'bonus1 corr:', corrbonus1,'\n bonus2 corr:',corrbonus2




if __name__ == "__main__":
    event_data = np.genfromtxt('events_US_air_traffic_GMT.txt', names = True, dtype=int)
    event_data.sort(order=['StartTime'])
    network = nx.read_weighted_edgelist('aggregated_US_air_traffic_network_undir.edg')
    Task1(event_data,network)
    Task2and3(event_data,network,task_number=2)
    Task4(event_data,network)
    Task5(event_data,network)
    Task6(event_data, network)

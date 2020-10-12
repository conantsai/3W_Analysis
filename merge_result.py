import numpy as np

def ProcessLinks(timeline, result_transpose, links, label):
    """[Pocess links for marking label of each timeline]

    Arguments:
        timeline {list} -- [Video's timeline]
        result_transpose {[type]} -- [Predict results through the model(Timeline is the horizontal axis and class is the vertical axis)]
        links {[type]} -- [links between nodes]
        label {list} -- [Marking results of each timeline]

    Returns:
        [type] -- [Labeling result]
    """    
    ## Sort the link from long to short, and put the same length(Based on timeline (horizontal)) in the same list.
    links_sort = sorted(links, key = lambda i:len(i), reverse=True)
    links_sort2 = [[links_sort[0]]]

    for i in range(1, len(links_sort)):
        ## If same length put in same list.
        if len(links_sort[i]) == len(links_sort[i-1]):
            links_sort2[-1].append(links_sort[i])
        ## If different length create new list and put it in.
        elif len(links_sort[i]) < len(links_sort[i-1]):
            links_sort2.append([links_sort[i]])

    ## Start processing from the longest link.
    for links in links_sort2:
        ## Mark the timeline if the link only have one.
        if len(links) == 1:
            for node in links[0]:
                if label[node[1]] is None: label[node[1]] = node[0]
        ## If the link not just have one.
        else:
            ## Find overlap
            overlap = [[] for i in range(len(timeline))]
            for links_index in range(len(links)):
                for pair_index in range(len(links[links_index])):
                    for compare_links_index in range(links_index, len(links)):
                        for compare_pair_index in range(len(links[compare_links_index])):
                            ## If two nodes are the same node.
                            if links[links_index][pair_index] == links[compare_links_index][compare_pair_index]:
                                break
                            else:
                                ## If two nodes is overlap.
                                if links[links_index][pair_index][1] == links[compare_links_index][compare_pair_index][1]:
                                    if links[links_index][pair_index] not in overlap[links[links_index][pair_index][1]]:
                                        overlap[links[links_index][pair_index][1]].append(links[links_index][pair_index])
                                    if links[compare_links_index][compare_pair_index] not in overlap[links[compare_links_index][compare_pair_index][1]]:
                                        overlap[links[compare_links_index][compare_pair_index][1]].append(links[compare_links_index][compare_pair_index])
            
            ## Mark timeline if node not overlapping.
            all_links_node = [ j for i in links for j in i]
            all_overlap_node = [ j for i in overlap for j in i]

            for node in all_links_node:
                if node not in all_overlap_node:
                    if label[node[1]] is None: label[node[1]] = node[0]
                    
            ## Sort the overlap link from long to short.
            overlap = [x for x in overlap if x != []]
            overlap_sort = sorted(overlap, key = lambda i:len(i), reverse=True)    

            ## If have overlapping.
            if len(overlap) != 0 :
                ## Put the same length(Based on class (vertical)) in the same list.
                overlap2 = [[overlap_sort[0]]]
                for i in range(1, len(overlap_sort)):
                    if len(overlap_sort[i]) == len(overlap_sort[i-1]):
                        overlap2[-1].append(overlap_sort[i])
                    elif len(overlap_sort[i]) < len(overlap_sort[i-1]):
                        overlap2.append([overlap_sort[i]])
                
                
                ## Caculate sum of the nodes's score.
                for overlap in overlap2:
                    for overlap_index in range(len(overlap)):
                        if overlap_index == 0: 
                            sum = [[] for i in range(len(overlap[overlap_index]))]
                            for i in range(len(overlap[overlap_index])):
                                sum[i] = overlap[overlap_index][i][2]
                        else:
                            if (overlap[overlap_index][0][1] -1) == overlap[overlap_index - 1][0][1]:
                                for i in range(len(overlap[overlap_index])):
                                    sum[i] = sum[i] + overlap[overlap_index][i][2]
                
                    ## Calculate the maximum number.
                    max_count = [i for i in range(len(sum)) if sum[i] == max(sum)]

                    ## If max score of overlap link only one, mark timeline .
                    if len(max_count) == 1:
                        for i in range(len(overlap)):
                            if label[overlap[i][max_count[0]][1]] is None: label[overlap[i][max_count[0]][1]] = overlap[i][max_count[0]][0]
                    ## If max score of overlap link not only one, add the neighbor's node score for overlapping score's sum.
                    else:
                        max_count2 = max_count
                        extractadd_front = 1
                        extractadd_back = 1

                        ## Until only one sum that is the maximum or add to the end.
                        while((len(max_count2) == len(max_count)) and ((overlap[0][0][1] - extractadd_front) >= 0) and ((overlap[-1][0][1] + extractadd_back) < len(timeline))):

                            ## If overlap's fist node is the first timeline.
                            if (overlap[0][0][1] - extractadd_back) == 0:
                                for i in max_count:
                                    sum[i] = sum[i] + result_transpose[overlap[-1][i][0]][overlap[-1][i][1] + extractadd_back]
                                extractadd_back += 1
                            ## If overlap's last node is the last timeline.
                            elif (overlap[-1][0][1] + extractadd_front) == (len(timeline) - 1):
                                for i in max_count:
                                    sum[i] = sum[i] + result_transpose[overlap[0][i][0]][overlap[0][i][1] - extractadd_front] 
                                extractadd_front += 1
                            else: 
                                for i in max_count:
                                    sum[i] = sum[i] + result_transpose[overlap[0][i][0]][overlap[0][i][1]-extractadd_front] + result_transpose[overlap[-1][i][0]][overlap[-1][i][1]+extractadd_back]

                                extractadd_back += 1
                                extractadd_front += 1

                            ## Calculate the maximum number.
                            max_count2 = [i for i in range(len(sum)) if sum[i] == max(sum)]
                        
                        ## If still can’t find the only maximum's sum, directly mark the first links for timeline.
                        if (len(max_count2) == len(max_count)):
                            for i in range(len(overlap)):
                                if label[overlap[i][max_count2[0]][1]] is None: label[overlap[i][max_count2[0]][1]] = overlap[i][max_count2[0]][0]
                        else:
                            ## Mark timeline.
                            for i in range(len(overlap)):
                                if label[overlap[i][max_count2[0]][1]] is None: label[overlap[i][max_count2[0]][1]] = overlap[i][max_count2[0]][0]

    return label

def ReconnectLink(timeline, label, ori_links):
    """[Remove the marked nodes from the links and reconnect the links]

    Arguments:
        timeline {list} -- [Video's timeline]
        label {list} -- [Marking results of each timeline]
        ori_links {[type]} -- [Original links]

    Returns:
        [type] -- [description]
    """    
    ## Find node that has not been marked.
    unmark_node = list()
    for index, mark in enumerate(label):
        if mark is None :
            ## Because the link is connected to the previous node, the previous node also need to retain.
            if ((index - 1) not in unmark_node) and ((index - 1) >= 0):
                unmark_node.append(index-1)
            
            if (index not in unmark_node): 
                unmark_node.append(index)
            
            ## Because the link is connected to the next node, the next node also need to retain.
            if ((index + 1) not in unmark_node) and ((index + 1) < len(timeline)):
                unmark_node.append(index+1)

    ## Deleting the marked nodes, and reconnect link.
    re_links = list()
    for link in ori_links:
        re_links.append([])
        for node in link:      
            if node[1] not in unmark_node:
                re_links.append([])
            else:
                re_links[-1].append(node)
    re_links = [x for x in re_links if x != []]
    
    return re_links

def ClassMerge(prediction_result, pool_a, pool_b, pool_c) -> list:
    """[Combine the result of the neighbor's timeline to determine the result of the current timeline]

    Arguments:
        prediction_result {2-dims list} -- [Predict results through the model((Class is the horizontal axis and timeline is the vertical axis))]

    Returns:
        list -- [Marked results after processing at each timeline]
    """    
    ## Transpose the score array.
    result_transpose = np.transpose(prediction_result)
    
    ## Initial the mark list.
    timeline = [i for i in range(result_transpose.shape[1])]
    label = [None for i in range(len(timeline))]  

    red = list()
    yellow = list()
    blue = list()

    ## Build the link(red, yellow, blue).
    for i in range(result_transpose.shape[0]):
        pre = ""
        for j in range(1, result_transpose.shape[1]):
            ## Build a definition link (red, yellow, blue) between two nodes.
            if result_transpose[i][j] in (pool_a):
                if result_transpose[i][j-1] in (pool_a):
                    now = "red"
                elif result_transpose[i][j-1] in (pool_b):
                    now = "yellow"
            elif result_transpose[i][j] in (pool_b):
                if result_transpose[i][j-1] in (pool_a):
                    now = "yellow"
                elif result_transpose[i][j-1] in (pool_b):
                    now = "blue"
                elif result_transpose[i][j-1] in (pool_c):
                    now = "white"
            elif result_transpose[i][j] in (pool_c):
                now = "white"

            ## Connect two links.
            if now != pre:
                if now == "red":
                    red.append([[i, j-1, result_transpose[i][j-1]], [i, j, result_transpose[i][j]]])
                elif now == "yellow":
                    yellow.append([[i, j-1, result_transpose[i][j-1]], [i, j, result_transpose[i][j]]])
                elif now == "blue":
                    blue.append([[i, j-1, result_transpose[i][j-1]], [i, j, result_transpose[i][j]]])
            elif now == pre:
                if now == "red":
                    red[-1].append([i, j, result_transpose[i][j]])
                elif now == "yellow":
                    yellow[-1].append([i, j, result_transpose[i][j]])
                elif now == "blue":
                    blue[-1].append([i, j, result_transpose[i][j]])

            pre = now

    ## Process red links.
    label = ProcessLinks(timeline, result_transpose, red, label)

    ## If all marked.
    if None not in label:
        return label

    ## Process yellow links.
    re_yellow = ReconnectLink(timeline, label, yellow)

    if len(re_yellow) != 0:
        label = ProcessLinks(timeline, result_transpose, re_yellow, label)
     
    ## If all marked.
    if None not in label:
        return label

    ## Process blue links.
    re_blue = ReconnectLink(timeline, label, blue)

    if len(re_blue) != 0:
        label = ProcessLinks(timeline, result_transpose, re_blue, label)

    ## If all marked.
    if None not in label:
        return label

    ## The remaining unmark nodess directly mark class with the largest prediction result.
    for index, content in enumerate(label):
        if content is None:
            label[index] = prediction_result[index].tolist().index(max(prediction_result[index]))

    return label

if __name__ == "__main__":
    pool_a = [7, 6]
    pool_b = [5, 4, 3]
    pool_c = [2, 1]

    prediction_result = np.asarray([[7, 6, 5, 4, 3, 2, 1],
                                    [7, 6, 4, 2, 3, 1, 5],
                                    [4, 6, 7, 5, 1, 2, 3],
                                    [6, 7, 5, 3, 1, 4, 2],
                                    [3, 4, 7, 6, 5, 2, 1],
                                    [2, 3, 4, 5, 6, 7, 1],
                                    [3, 4, 7, 5, 2, 1, 6],
                                    [4, 3, 7, 5, 6, 2, 1],
                                    [1, 4, 5, 7, 6, 3, 2],
                                    [7, 5, 3, 4, 2, 1, 6]])
    
    label = ClassMerge(prediction_result, pool_a, pool_b, pool_c)
    print(label)
    
    pool_a = [2]
    pool_b = [1]
    pool_c = []

    prediction_result = np.asarray([[2, 1],
                                    [2, 1],
                                    [2, 1],
                                    [1, 2],
                                    [1, 2],
                                    [2, 1],
                                    [2, 1],
                                    [2, 1],
                                    [1, 2],
                                    [1, 2]])
    
    label = ClassMerge(prediction_result, pool_a, pool_b, pool_c)

    print(label)
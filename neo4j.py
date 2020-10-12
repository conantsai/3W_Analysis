from py2neo import *
import pandas as pd
import os

graph =  Graph("http://127.0.0.1:7474",username="neo4j",password="uscc")

def create_graph():
    morning = Node("WhenState", name="morning")
    afternoon = Node("WhenState", name="afternoon")
    evening = Node("WhenState", name="evening")
    midnight = Node("WhenState", name="midnight")

    work = Node("WhereState", name="laboratory")
    rest = Node("WhereState", name="bedroom")

    computer = Node("WhatState", name="computer")
    phone = Node("WhatState", name="phone")
    bowl = Node("WhatState", name="bowl")
    bed = Node("WhatState", name="bed")
    pen = Node("WhatState", name="pen")
    book = Node("WhatState", name="book")
    NoObject = Node("WhatState", name="NoObject")

    when_t = (morning, afternoon, evening, midnight)
    where_t = (work, rest)
    what_t = (computer, phone, bowl, bed, pen, book, NoObject)

    main_state_count = 0

    for i in what_t:
        for j in where_t:
            relationship = Relationship(i, i["name"], j)
            graph.create(relationship)
            for k in when_t:
                relationship = Relationship(j, i["name"] + "_" + j["name"], k)
                graph.create(relationship)

                main = Node("MainState", name=str(main_state_count))
                relationship = Relationship(k, i["name"] + "_" + j["name"] + "_" + k["name"], main)
                graph.create(relationship)

                main_state_count += 1

    # node_matcher = NodeMatcher(graph)

    # node2 = list(node_matcher.match('MainState').where("_.name = '2'"))[0]
    # node1 = list(node_matcher.match('MainState').where("_.name = '1'"))[0]
    # node3 = list(node_matcher.match('MainState').where("_.name = '3'"))[0]
    # relationship = Relationship(node2, "score", node1, score=1)
    # graph.create(relationship)
    # relationship = Relationship(node2, "score", node3, score=1)
    # graph.create(relationship)

def update_relationship(cur, pre, graph):
    relationship_matcher = RelationshipMatcher(graph)
    
    cur_relationship = graph.match(r_type=cur)
    pre_relationship = graph.match(r_type=pre)

    print(list(cur_relationship), list(pre_relationship))

    cur_node = list(cur_relationship)[0].end_node
    pre_node = list(pre_relationship)[0].end_node
    
    relationship = list(relationship_matcher.match({pre_node, cur_node}))

    if len(relationship) == 0:
        new_relationship = Relationship(pre_node, "score", cur_node, pos_score=1, neg_score=0)
        graph.create(new_relationship)
    elif len(relationship) == 1:
        if relationship[0].start_node["name"] == pre_node["name"]:
            relationship_score = int(relationship[0]["pos_score"])
            relationship[0]["pos_score"] = relationship_score + 1 
            graph.push(relationship[0])
        else:
            relationship_score = int(relationship[0]["neg_score"])
            relationship[0]["neg_score"] = relationship_score + 1 
            graph.push(relationship[0])

def search_action(what, when, where, graph):
    node_matcher = NodeMatcher(graph)
    relationship_matcher = RelationshipMatcher(graph)
    
    max_relationship = [0, ""]

    ## Wheather have this what node
    what_node = list(node_matcher.match('WhatState').where("_.name = " + "\'" + what + "\'"))
    if len(what_node) != 0:
        ## Find the nodes related with this what node
        what_relationship = list(relationship_matcher.match({what_node[0]}, what))

        for i in range(len(what_relationship)):
            ## Wheather have where node connected to this what node
            if what_relationship[i].end_node["name"] == where:
                where_node = list(node_matcher.match('WhereState').where("_.name = " + "\'" + where + "\'"))
                ## Find the nodes related with this where node
                where_relationship = list(relationship_matcher.match({where_node[0]}, what + "_" + where)) 
                
                for j in range(len(where_relationship)):
                    ## Wheather have when node connected to this where node
                    if where_relationship[j].end_node["name"] == when:
                        when_node = list(node_matcher.match('WhenState').where("_.name = " + "\'" + when + "\'"))
                        ## Find the nodes related with this when node
                        when_relationship = list(relationship_matcher.match({when_node[0]}, what + "_" + where + "_" + when))
                        if len(when_relationship) != 0:
                            main_state = when_relationship[0].end_node
                            main_relationship = list(relationship_matcher.match({main_state}))
                            print(main_relationship)
                            if len(main_relationship) != 0:
                                for k in range(len(main_relationship)):
                                    if main_relationship[k].start_node == main_state:
                                        if int(main_relationship[k]["score"]) > max_relationship[0]:
                                            max_relationship = [int(main_relationship[k]["score"]), main_relationship[k].end_node["name"]]
                        else:
                            pass
                    if j == len(where_relationship)-1:
                        pass
            if i == len(what_relationship)-1:
                pass

    if max_relationship[0] == 1:
        main_node = list(node_matcher.match('MainState').where("_.name = " + "\'" + max_relationship[1] + "\'"))
        main_relationship = list(relationship_matcher.match({main_node[0]})) 
        for i in main_relationship:
            if (type(i).__name__) != "score":
                next_behavior = type(i).__name__
    else:
        next_behavior = "None"
    return(next_behavior)


if __name__ == "__main__":
    # Delelte default lnowledge graph
    graph.delete_all()

    ## Create default lnowledge graph
    create_graph()

    ## Create relationship from past record
    for root, dirs, files in os.walk("record"):
        for file_name in files:
            file = pd.read_csv(os.path.join(root, file_name))

            for content in range(file.shape[0]):
                if content == 0:
                    pass
                else:
                    cur = file.iloc[content].what_state + "_" + file.iloc[content].where_state + "_" + file.iloc[content].when_state
                    pre = file.iloc[content-1].what_state + "_" + file.iloc[content-1].where_state + "_" + file.iloc[content-1].when_state
                    
                    update_relationship(cur=cur, pre=pre, graph=graph)

import json

def dict_to_json(dict, json_path):
    with open(json_path,"w", encoding="utf8") as f:
        f_json = json.dumps(speaking_dict, ensure_ascii=False)
        f.write(f_json)

if __name__ == "__main__":
    pos_dict = {"n": 0, 
                "nr": 1, 
                "nrfg *": 2,
                "nt": 3,
                "nz": 4,
                "ns": 5,
                "tg": 6,
                "v": 7,
                "df *": 8,
                "vg": 9,
                "d": 10,
                "t": 11}

    speaking_dict = {"n": ["委購案",
                           "學合案",
                           "床墊",
                           "智慧床墊"], 
                     "nr": ["昀陽", 
                            "之彧", 
                            "昌祐", 
                            "子源", 
                            "崇智", 
                            "亦心", 
                            "小豪", 
                            "威智",
                            "佳萱",
                            "在偉",
                            "郁淞",
                            "登立",
                            "育雯"], 
                     "nrfg *": ["李昀陽",
                                "蘇之彧",
                                "何昌祐",
                                "王子元",
                                "潘崇智",
                                "田亦心",
                                "黃威智",
                                "林佳萱",
                                "徐郁淞",
                                "王登立"],
                     "nt": ["中科院"],
                     "nz": [],
                     "ns": ["活力小廚",
                            "奧利薇",
                            "大將",
                            "七海魚皮",
                            "菇雞",
                            "鄭記肉夾饃",
                            "阿偉火雞肉飯",
                            "舞村",
                            "廣越",
                            "九湯屋",
                            "肉控肉"],
                     "tg": [],
                     "v": [],
                     "df *": [],
                     "vg": [],
                     "d": [],
                     "t": []}

    dict_to_json(dict=speaking_dict, json_path="nlp_recognize/speak_focus.json")
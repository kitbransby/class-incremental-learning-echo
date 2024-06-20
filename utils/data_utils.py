def class_maps(model_id):
    "model ids: wase 0, camus 1, mstr 2, stg 3, mahi 4, uoc 5"
    if model_id == 0:
        map_ = {0:4, 1:5, 2:6, 3:7, 4:8, 5:9, 6:10, 7:11}
    elif model_id == 1:
        map_ = {0:4, 1:6}
    elif model_id == 2:
        map_ = {0:4, 1:5, 2:6, 3:7, 4:8, 5:9, 6:10, 7:11, 8:12, 9:13, 10:14}
    elif model_id == 3:
        map_ = {0:0, 1:1, 2:2, 3:3}
    elif model_id == 4:
        map_ = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14}
    elif model_id == 5:
        map_ = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14}
    else:
        print('no map available')
    return map_
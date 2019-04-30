from ..data.entity import User, Item, Event
import numpy as np

from sklearn.utils import Bunch

def csv_loader(file):
    u_dict = {}
    i_dict = {}
    samples = []
    user_counter = 0
    item_counter = 0

    with open(file, 'r') as f:
        for line in f:
            line.replace(" ", "")
            line.replace("\n", "")
            if line == "":
                continue
            (user_id, item_id, rating, timestamp) = line.split(',')
            if user_id in u_dict:
                user = u_dict[user_id]
            else:
                user = User(user_counter)
                user_counter += 1
                u_dict[user_id] = user

            if item_id in i_dict:
                item = i_dict[item_id]
            else:
                item = Item(item_counter)
                item_counter += 1
                i_dict[item_id] = item

            sample = Event(user, item, int(rating))
            samples.append(sample)

    return Bunch(samples=samples,
                 n_user=len(u_dict),
                 n_item=len(i_dict),
                 n_sample=len(samples))

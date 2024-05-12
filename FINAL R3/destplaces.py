def shift_dest_to_end(lst, dest_places):
    dest_places_set = set(dest_places)
    without_dest = [item for item in lst if item not in dest_places_set]
    with_dest = [item for item in lst if item in dest_places_set]
    return without_dest + with_dest

places = ["gold","wood", "furnace"]
dest_places = ["gold", "furnace"]

new_order = shift_dest_to_end(places, dest_places)
print(new_order)

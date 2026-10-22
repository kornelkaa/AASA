from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-mini", dataroot="./data/nuscenes", verbose=True)
my_scene = nusc.scene[0]

print("Sikeres betöltés, első scene leírása", my_scene["description"])


my_sample = nusc.sample[10]
nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
nusc.render_sample_data(my_sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True)
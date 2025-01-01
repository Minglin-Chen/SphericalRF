def parse_cameras(cameras):
    assert len(cameras) == 1
    camera = list(cameras.values())[0]

    camera_model = camera.model
    width, height = camera.width, camera.height
    params = camera.params

    fx = fy = None
    cx = cy = None
    k1 = k2 = None
    p1 = p2 = None
    if camera_model == 'SIMPLE_PINHOLE':
        fx, cx, cy = params
    elif camera_model == 'PINHOLE':
        fx, fy, cx, cy = params
    elif camera_model == 'SIMPLE_RADIAL':
        fx, cx, cy, k1 = params
    elif camera_model == 'RADIAL':
        fx, cx, cy, k1, k2 = params
    elif camera_model == 'OPENCV':
        fx, fy, cx, cy, k1, k2, p1, p2 = params
    else:
        raise NotImplementedError

    return width, height, fx, fy, cx, cy, k1, k2, p1, p2
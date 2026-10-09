"""Existing diagnostic functions, copied without AST changes from the executed sources.
See docs/provenance.md and scripts/original for origins. No new method.
"""

import numpy as np

from tcaff.realign.object_map import ObjectMap

from tcaff.fastsam3D.depth_mask_2_centroid import depth_mask_2_centroid, mask_depth_2_width_height

def ideal_view(dimensions, degrees, K, shape, distance):
    """Ray/OBB front-surface intersections, in camera RDF coordinates."""
    y, x = np.indices(shape)
    rays = np.stack(((x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones(shape)), axis=-1)
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    center = np.array([0.,0.,distance])
    # dimensions are horizontal side, depth side, vertical height.
    half = np.array([dimensions[0],dimensions[2],dimensions[1]])/2
    origin = -rotation.T@center
    local_rays = rays@rotation
    with np.errstate(divide='ignore',invalid='ignore'):
        t0 = (-half-origin)/local_rays
        t1 = (half-origin)/local_rays
    near = np.max(np.minimum(t0,t1),axis=-1)
    far = np.min(np.maximum(t0,t1),axis=-1)
    valid = (near>0)&np.isfinite(near)&(near<=far)
    mask = valid.astype(np.uint8)
    depth = np.where(valid,near*1000.,0.)
    assert np.any(valid) and not np.any(valid[[0,-1],:]) and not np.any(valid[:,[0,-1]])
    ys, xs = np.nonzero(mask)
    center_pixel = (float(np.mean(xs)),float(np.mean(ys)))
    centroid = depth_mask_2_centroid(depth,mask,center_pixel,K)
    width, height = mask_depth_2_width_height(centroid[2],mask,K)
    np.testing.assert_allclose([width,height],[(xs.max()-xs.min())*centroid[2]/K[0,0],
                                               (ys.max()-ys.min())*centroid[2]/K[1,1]],atol=1e-14,rtol=0.)
    return dict(angle_deg=degrees, width_m=float(width),height_m=float(height),
                median_depth_m=float(centroid[2]), mask_pixels=int(valid.sum()),
                map_row=[*centroid.tolist(),float(width),float(height),1.]),mask,depth

def hidden_side(mask, fraction, axis):
    """Hide the left/top fraction of the silhouette bounding-box extent."""
    assert axis in ['left', 'top'] and 0 <= fraction < 1
    ys, xs = np.nonzero(mask)
    coordinates = xs if axis == 'left' else ys
    low, high = int(coordinates.min()), int(coordinates.max())
    removed = int(round(fraction * (high-low+1)))
    visible = mask.copy()
    if axis == 'left':
        visible[:, :low+removed] = 0
    else:
        visible[:low+removed, :] = 0
    assert np.any(visible) and np.all(visible <= mask)
    return visible

def describe(mask, depth, K):
    ys, xs = np.nonzero(mask)
    mean = [float(xs.mean()), float(ys.mean())]
    center = depth_mask_2_centroid(depth, mask, mean, K)
    width, height = mask_depth_2_width_height(center[2], mask, K)
    assert np.all(np.isfinite(center)) and width > 0 and height > 0
    return dict(visible_pixels=int(mask.sum()), pixel_mean=mean, centroid_camera_RDF_m=center.tolist(),
                width_m=float(width), height_m=float(height), map_row=[*center.tolist(), float(width), float(height), 1.])

def object_map(rows):
    a = np.asarray(rows, float).reshape(-1, 6)
    return ObjectMap(a[:, :3], a[:, 3], a[:, 4], a[:, 5])

def gate_details(a, b, ratio=1.35, diff=.1):
    checks = dict(width_ratio=max(a[3], b[3]) > min(a[3], b[3]) * ratio,
                  height_ratio=max(a[4], b[4]) > min(a[4], b[4]) * ratio,
                  width_difference=abs(a[3]-b[3]) > diff,
                  height_difference=abs(a[4]-b[4]) > diff)
    failed = [k for k, v in checks.items() if bool(v)]
    return dict(admitted=not failed, failed_checks=failed,
                first_exclusion_reason=failed[0] if failed else None,
                width_difference_m=float(abs(a[3]-b[3])), height_difference_m=float(abs(a[4]-b[4])),
                ratio_limit=ratio, absolute_difference_limit_m=diff)

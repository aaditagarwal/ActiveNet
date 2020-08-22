import math
import numpy as np

# def find_angle(joint1, joint2):
#     '''
#     joint1.shape = (2,)
#     joint2.shape = (2,)
#     Returns angle between the joints
#     '''
#     angle = math.atan2(joint2[1] - joint1[1], joint2[0] - joint1[0])
#     return angle

def find_angle_between_three_kpoints(kp1, kp2, kp3):
    '''
    kp1, kp2, kp3 shape = (2,) each
    '''
    try:
        assert (not np.equal(list(kp1), list(kp2)).all()) and (not np.equal(list(kp2), list(kp3)).all())
    except:
        pass
    
    a = np.asarray(kp1)
    b = np.asarray(kp2)
    c = np.asarray(kp3) 

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
def generate_angles_between_joints(joints):
    '''
    joints.shape = (17,2)
    '''
    nose_reye_rear = find_angle_between_three_kpoints(joints[0], joints[14], joints[16])
    nose_leye_lear = find_angle_between_three_kpoints(joints[0], joints[15], joints[17])

    neck_nose_rear = find_angle_between_three_kpoints(joints[1], joints[0], joints[16])
    neck_nose_lear = find_angle_between_three_kpoints(joints[1], joints[0], joints[17])

    lsh_neck_nose = find_angle_between_three_kpoints(joints[5], joints[1], joints[0])
    rsh_neck_nose = find_angle_between_three_kpoints(joints[2], joints[1], joints[0])

    neck_rsh_relb = find_angle_between_three_kpoints(joints[1], joints[2], joints[3])
    neck_lsh_lelb = find_angle_between_three_kpoints(joints[1], joints[5], joints[6])

    rsh_relb_rwr = find_angle_between_three_kpoints(joints[2], joints[3], joints[4])
    lsh_lelb_lwr = find_angle_between_three_kpoints(joints[5], joints[6], joints[7])

    core_joint = (joints[11] + joints[8]) / 2.0
    core_neck_nose = find_angle_between_three_kpoints(joints[0], joints[1], core_joint)

    neck_core_rhip = find_angle_between_three_kpoints(joints[1], core_joint, joints[8])
    neck_core_lhip = find_angle_between_three_kpoints(joints[1], core_joint, joints[11])

    rhip_rknee_rankle = find_angle_between_three_kpoints(joints[8], joints[9], joints[10])
    lhip_lknee_lankle = find_angle_between_three_kpoints(joints[11], joints[12], joints[13])

    return [nose_reye_rear, nose_leye_lear, neck_nose_rear, neck_nose_lear, lsh_neck_nose,
            rsh_neck_nose, neck_rsh_relb, neck_lsh_lelb, rsh_relb_rwr, lsh_lelb_lwr, core_neck_nose,
            neck_core_rhip, neck_core_lhip, rhip_rknee_rankle, lhip_lknee_lankle]
    


# print(find_angle_between_three_kpoints((267,275), (267,275), (231,250)))



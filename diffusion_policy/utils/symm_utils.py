import torch

def get_joint_permutation_matrix(joint_names):
    # Create two lists, one for left body parts and one for right body parts
    left_names = [name for name in joint_names if name.startswith('left')]
    right_names = [name for name in joint_names if name.startswith('right')]
    
    # Check if the number of left and right body parts are equal
    if len(left_names) != len(right_names):
        raise ValueError("The number of left and right body parts must be equal for permutation.")
    
    matrix = torch.arange(len(joint_names))

    # Create a mapping from left body parts to right body parts
    for i, left_name in enumerate(left_names):
        left_index = joint_names.index(left_name)
        right_name = 'right' + left_name.split('left')[1]
        
        if right_name in joint_names:
            right_index = joint_names.index(right_name)
            matrix[left_index] = right_index
            matrix[right_index] = left_index
    return matrix

def get_joint_reflection_matrix(joint_names):
    matrix = torch.ones(len(joint_names))

    # Create a mapping from left body parts to right body parts
    for i, name in enumerate(joint_names):
        if 'roll' in name or 'yaw' in name:
            matrix[i] = -1
    return matrix

def get_body_permutation_matrix(body_names):
    # Create two lists, one for left body parts and one for right body parts
    left_names = [name for name in body_names if name.startswith('left')]
    right_names = [name for name in body_names if name.startswith('right')]
    
    # Check if the number of left and right body parts are equal
    if len(left_names) != len(right_names):
        raise ValueError("The number of left and right body parts must be equal for permutation.")
    
    matrix = torch.arange(len(body_names))

    # Create a mapping from left body parts to right body parts
    for i, left_name in enumerate(left_names):
        left_index = body_names.index(left_name)
        right_name = 'right' + left_name.split('left')[1]
        
        if right_name in body_names:
            right_index = body_names.index(right_name)
            matrix[left_index] = right_index
            matrix[right_index] = left_index
    return matrix

def get_reflect_op(reps):
        reps_shape = []
        for i in range(len(reps)):
            assert reps[i].shape[1] == reps[i].shape[0]
            reps_shape.append(reps[i].shape[0])

        reflect_op = torch.zeros((sum(reps_shape), sum(reps_shape)))

        for i in range(len(reps)):
            idx0 = sum(reps_shape[:i])
            idx1 = sum(reps_shape[:i + 1])
            reflect_op[idx0:idx1, idx0:idx1] = reps[i]

        return reflect_op

def get_reflect_reps(body_names, joint_names):
    Rd = torch.eye(3)
    Rd[1, 1] = -1
    Rd_pseudo = torch.eye(3)
    Rd_pseudo[[0, 2], [0, 2]] = -1

    jperm = get_joint_permutation_matrix(joint_names)
    jref = get_joint_reflection_matrix(joint_names)
    Q = torch.zeros((len(jperm), len(jperm)))
    Q[torch.arange(len(jperm)), jperm] = jref

    bperm = get_body_permutation_matrix(body_names)
    Q_Rd = torch.zeros((len(bperm), len(bperm), 3, 3))
    Q_Rd[torch.arange(len(bperm)), bperm] = Rd[None,:,:].repeat(len(bperm), 1, 1)
    Q_Rd = Q_Rd.permute(0,2,1,3).reshape(len(bperm)*3,len(bperm)*3)
    Q_Rd_pseudo = torch.zeros((len(bperm), len(bperm), 3, 3))
    Q_Rd_pseudo[torch.arange(len(bperm)), bperm] = Rd_pseudo[None,:,:].repeat(len(bperm), 1, 1)
    Q_Rd_pseudo = Q_Rd_pseudo.permute(0,2,1,3).reshape(len(bperm)*3,len(bperm)*3)
    return Q, Rd, Rd_pseudo, Q_Rd, Q_Rd_pseudo, len(bperm)

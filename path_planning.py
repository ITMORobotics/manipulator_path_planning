from cmath import nan
import roboticstoolbox as rtb
import spatialmath as spm
import spatialmath.base as spmb
import numpy as np
import matplotlib.pyplot as plt

np.printoptions(precision=2, suppress=True)


def calculate_quaternion_matrix(q: np.ndarray) -> np.ndarray:
    # See E matrix in https://arxiv.org/pdf/0811.2889.pdf
    qe = np.array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]]
    ])

    return qe

def get_error(goal: spm.SE3, transform: spm.SE3) -> np.ndarray:
    # as exapmle: https://robotics.stackexchange.com/questions/6617/jacobian-based-trajectory-following
    error_translation = goal.t - transform.t
    qe = calculate_quaternion_matrix(spm.UnitQuaternion(transform).A)
    return np.concatenate((
        error_translation, 2*qe @ spm.UnitQuaternion(goal).A
    ))

####################################################################### MAIN PART

# DH parameters
d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
a = np.array([0, -0.425, -0.3922, 0, 0, 0])
alpha = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
offset = np.array([0, 0, 0, 0, 0, 0])

# q_lim_bottom = -np.ones(6)*2*np.pi
# q_lim_top = np.ones(6)*2*np.pi

links = [rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i]) for i in range(6)]
ur_robot = rtb.DHRobot(links, name='UR5e')

def Tr(p_star, R_star, q):
    goal = spm.SE3(spmb.rt2tr(R_star, p_star))
    gamma = 1
    stop = 1e-10

    q_lim_bottom = -np.ones(6)*2*np.pi
    q_lim_top = np.ones(6)*2*np.pi
    e = np.ones(6)
    i = 100
    while np.linalg.norm(e) > stop and i > 0:
        transform = ur_robot.fkine(q)
        e = get_error(goal, transform)

        J = ur_robot.jacob0(q)
        J_inv = np.linalg.pinv(J.T.dot(J)).dot(J.T)
        q = q + gamma*J_inv.dot(e)

        k = np.argwhere(q > q_lim_top)
        q[k] -= 2*np.pi

        k = np.argwhere(q < q_lim_bottom)
        q[k] += 2*np.pi
        # print(np.linalg.norm(e))
        print(i, np.linalg.norm(e), np.linalg.det(J))
        print(q)
        i -= 1

    k_b = [1]; k_t = [1]
    while len(k_b) > 0 or len(k_t) > 0:
        k_t = np.argwhere(q > q_lim_top)
        q[k_t] -= 2*np.pi

        k_b = np.argwhere(q < q_lim_bottom)
        q[k_b] += 2*np.pi

    return q, e

def Tr_transpose(p_star, R_star, q):
    goal = spm.SE3(spmb.rt2tr(R_star, p_star))
    stop = 1e-10

    q_lim_bottom = -np.ones(6)*2*np.pi
    q_lim_top = np.ones(6)*2*np.pi
    e = np.ones(6)

    i = 1000
    while np.linalg.norm(e) > stop and i > 0:
        transform = ur_robot.fkine(q)
        e = get_error(goal, transform)

        J = ur_robot.jacob0(q)
        delta = J @ J.T @ e
        alpha = 1
        if np.linalg.norm(e) > stop:
            alpha = (e.T @ delta)/(delta.T @ delta)

        q += alpha*J.T @ e

        k = np.argwhere(q > q_lim_top)
        q[k] -= 2*np.pi

        k = np.argwhere(q < q_lim_bottom)
        q[k] += 2*np.pi

        print(i, np.linalg.norm(e), np.linalg.det(J), alpha)
        i -= 1
    
    return q, e

def Tr_DLSQ(p_star, R_star, q):
    goal = spm.SE3(spmb.rt2tr(R_star, p_star))
    gamma = 0.5
    stop = 1e-10

    q_lim_bottom = -np.ones(6)*2*np.pi
    q_lim_top = np.ones(6)*2*np.pi
    e = np.ones(6)
    eps = 0.01
    detJ = []

    i = 1000
    while np.linalg.norm(e) > stop and i > 0:
        transform = ur_robot.fkine(q)
        e = get_error(goal, transform)

        J = ur_robot.jacob0(q)
        
        # d = np.linalg.det(J)
        # detJ.append(d)
        # print(np.linalg.norm(e))

        u, s, vh = np.linalg.svd(J)
        mins = min(s)
        if mins >= eps:
            lam = 0
        else:
            lam = (1 - (mins/eps)**2) * 0.00001

        J_inv = np.linalg.inv(J.T.dot(J) + lam * np.eye(6)).dot(J.T)
        q += gamma*J_inv.dot(e)

        k = np.argwhere(q > q_lim_top)
        q[k] -= 2*np.pi

        k = np.argwhere(q < q_lim_bottom)
        q[k] += 2*np.pi

        print(i, np.linalg.norm(e), np.linalg.det(J))
        i -= 1

    k_b = [1]; k_t = [1]
    while len(k_b) > 0 or len(k_t) > 0:
        k_t = np.argwhere(q > q_lim_top)
        q[k_t] -= 2*np.pi

        k_b = np.argwhere(q < q_lim_bottom)
        q[k_b] += 2*np.pi
    
    return q, e

def fabrik(p_star, R_star, q):
    p = ur_robot.fkine_path(q).t
    d = np.linalg.norm(p[1:,:] - p[:-1, :], axis=1)
    n = len(p) - 1
    jr = range(n-1, -1, 1)

    dist = np.linalg.norm(p[0] - p_star)
    if dist > np.sum(d):
        print('Target is unreachable')
    else:
        print('Target is reachable')
        b = np.copy(p[0])

        dif_A = np.linalg.norm(p[n] - p_star)
        while dif_A > 1e-10:

            # STAGE 1: FORWARD REACHING
            p[n] = np.copy(p_star)
            # i = n-1,...,1
            for i in jr:
                # Find the distance ri between the new joint position pi+1 and the joint pi
                r = np.linalg.norm(p[i+1] - p[i])
                lambd = d[i]/r
                # Find the new point positions pi
                p[i] = (1 - lambd)*p[i+1] + lambd*p[i]

            # STAGE 2: BACKWARD REACHING
            # Set the root pi its initial position
            p[0] = np.copy(b)
            # i = 1,...,n-1
            for i in np.flip(jr):
                # Find the distance ri between the new joint position pi and the joint pi+1
                r = np.linalg.norm(p[i+1] - p[i])
                lambd = d[i]/r
                # Find the new point positions pi
                p[i+1] = (1 - lambd)*p[i] + lambd*p[i+1]
            
            dif_A = np.linalg.norm(p[n] - p_star)
            print(dif_A)



def linpath(t1, t2):
    n = int(np.linalg.norm(t2 - t1)/0.0005)
    return np.linspace(t1, t2, n)


q_i = np.array([0, -90, 0, -90, 0, 0])*np.pi/180
q = np.array([0, -90, 0, -90, 0, 0])*np.pi/180
transform = ur_robot.fkine(q)

R_star = transform.R
t1 = transform.t

t2 = t1.copy()
t2[2] -= 0.05          # 5 cm down

t3 = t2.copy()
t3[0] -= 0.2

path1 = linpath(t1, t2)
path2 = linpath(t2, t3)

solution = []


# i = 5
# p_star = path1[i]
# fabrik(p_star, R_star, q_i)
print('PATH1, start loop, ' + str(path1.shape))
print(q)
for i in range(path1.shape[0]):
    print('i ' + str(i))
    p_star = path1[i]
    q, e = Tr_transpose(p_star, R_star, q_i)
    # q, e = Tr_DLSQ(p_star, R_star, q_i)
    # q, e = Tr(p_star, R_star, q_i)
    solution.append(np.concatenate((q, e)))
    q_i = np.copy(q)


solution = np.array(solution)
fig = plt.figure()
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.plot(solution[:, i-1])

plt.show()

fig = plt.figure()
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.plot(solution[:, i-1 + 6])

plt.show()
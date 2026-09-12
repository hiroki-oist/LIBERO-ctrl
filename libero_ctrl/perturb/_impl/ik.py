"""Damped least-squares IK in end-effector space.

Only the part needed at run time. It requires numpy and mujoco and nothing else, so that
applying a perturbation does not drag in torch or the calibration tooling.
"""
import numpy as np
import mujoco as _mj

def _axisangle2mat(v):
    a=np.linalg.norm(v)
    if a<1e-12: return np.eye(3)
    k=v/a; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*(K@K)

def _mat2axisangle(R):
    c=np.clip((np.trace(R)-1)/2,-1,1); a=np.arccos(c)
    if a<1e-9: return np.zeros(3)
    if abs(np.pi-a)<1e-6:
        w,V=np.linalg.eigh(R); k=np.real(V[:,np.argmin(np.abs(w-1))]); return a*k/np.linalg.norm(k)
    return a/(2*np.sin(a))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

def _grip_site(T): return T.m.site_name2id("gripper0_grip_site")

def eef_pose(T, st):
    T.env.set_init_state(st); T.sim.forward(); s=_grip_site(T)
    return T.sim.data.site_xpos[s].copy(), T.sim.data.site_xmat[s].copy().reshape(3,3)

def perturb_robot_eef(T, st, dpos, drot, iters=200, lam=0.08):
    """Solve for the joint angles that translate the end effector by dpos (m) and rotate it by
    drot (rad, axis-angle).

    Returns (new init state, position residual [m], orientation residual [deg], number of
    joints that hit a limit)."""
    m,sim=T.m,T.sim; s=_grip_site(T)
    p0,R0=eef_pose(T,st)
    pt=p0+np.asarray(dpos); Rt=_axisangle2mat(np.asarray(drot))@R0
    q=st[1:8].copy(); jacp=np.zeros((3,m.nv)); jacr=np.zeros((3,m.nv))
    for _ in range(iters):
        sim.data.qpos[0:7]=q; sim.forward()
        p=sim.data.site_xpos[s]; R=sim.data.site_xmat[s].reshape(3,3)
        ep=pt-p; er=_mat2axisangle(Rt@R.T)
        if np.linalg.norm(ep)<2e-4 and np.linalg.norm(er)<2e-3: break
        _mj.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, s)
        J=np.vstack([jacp[:,:7],jacr[:,:7]])
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6), np.concatenate([ep,er]))
        q=np.clip(q+dq, T.JR[:,0], T.JR[:,1])
    nclip=int(np.sum((np.abs(q-T.JR[:,0])<1e-6)|(np.abs(q-T.JR[:,1])<1e-6)))
    stn=st.copy(); stn[1:8]=q
    sim.data.qpos[0:7]=q; sim.forward()
    ep=np.linalg.norm(pt-sim.data.site_xpos[s])
    er=np.degrees(np.linalg.norm(_mat2axisangle(Rt@sim.data.site_xmat[s].reshape(3,3).T)))
    return stn, ep, er, nclip


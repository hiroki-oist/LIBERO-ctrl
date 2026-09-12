"""The LIBERO env wrapper (Task) and the deterministic env seed.

Everything needed at run time lives inside the package, so that nothing depends on sys.path
manipulation.
"""
import os, numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
UP=np.array([0.,0.,1.])

def quat2mat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def lookat_quat(p,t,up=UP):
    f=t-p; f/=np.linalg.norm(f); zc=-f
    xc=np.cross(up,zc); xc/=np.linalg.norm(xc); yc=np.cross(zc,xc)
    R=np.stack([xc,yc,zc],1); w=np.sqrt(max(1e-12,1+R[0,0]+R[1,1]+R[2,2]))/2
    return np.array([w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)])
def rot_z(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def rot_axis(ax,a):
    ax=ax/np.linalg.norm(ax); c,s=np.cos(a),np.sin(a)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)*c+s*K+(1-c)*np.outer(ax,ax)
def kelvin_rgb(T):
    t=T/100.
    r=255. if t<=66 else 329.698727446*((t-60)**-0.1332047592)
    g=99.4708025861*np.log(t)-161.1195681661 if t<=66 else 288.1221695283*((t-60)**-0.0755148492)
    b=255. if t>=66 else (0. if t<=19 else 138.5177312231*np.log(t-10)-305.0447927307)
    return np.clip(np.array([r,g,b]),0,255)/255.

def light_color(warm, tint):
    """Light colour, as an RGB gain that preserves Rec.709 luminance.

    warm : R x2^warm, B x2^-warm      (positive = warmer, negative = cooler)
    tint : G x2^tint, R,B x2^(-tint/2) (positive = green, negative = magenta)
    """
    c = np.array([2.0 ** warm, 1.0, 2.0 ** -warm], float)
    c = c * np.array([2.0 ** (-tint / 2), 2.0 ** tint, 2.0 ** (-tint / 2)])
    return c / (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def equiv_cct(warm, lo=1000., hi=40000.):
    """The blackbody colour temperature (K) closest to a given warm component. For reporting
    only -- the axis itself is not parameterised along the blackbody locus."""
    t = np.exp(np.linspace(np.log(lo), np.log(hi), 400))
    tgt = light_color(warm, 0.0)
    best = min(t, key=lambda x: float(np.sum((kelvin_rgb(x) / (0.2126 * kelvin_rgb(x)[0]
               + 0.7152 * kelvin_rgb(x)[1] + 0.0722 * kelvin_rgb(x)[2]) - tgt) ** 2)))
    return float(best)


ENV_SEED = 20260904      # pins the placement sampler; must match manifest.protocol.env_seed


def seed_env(seed=ENV_SEED):
    """Call immediately *before* constructing an env, to make fixture placement reproducible.

    LIBERO's 92-dimensional sim state does not contain the placement of static bodies. It holds
    qpos/qvel for the movable objects only; shelves, stoves and cabinets are positioned by
    robosuite's placement sampler when the env is constructed, and that sampler is not seeded.
    So the same task with the same init_id puts the fixtures somewhere slightly different on
    every run (measured on libero_spatial task 7: the wooden cabinet moves 8.5 mm and rotates,
    the flat stove moves 7.6 mm; see calibration/fixture_jitter.json). set_init_state does not
    bring them back, because they are not in the state.

    Nothing about this benchmark reproduces if this is not pinned.
    """
    import random
    random.seed(seed); np.random.seed(seed)
    return seed


class Task:
    def __init__(self, suite, ti, H=256, W=256, seed=ENV_SEED):
        if seed is not None: seed_env(seed)
        self.seed=seed
        bm=benchmark.get_benchmark_dict()[suite]()
        self.task=bm.get_task(ti); self.ti=ti; self.H,self.W=H,W
        self.suite_name=suite
        bddl=os.path.join(get_libero_path("bddl_files"),self.task.problem_folder,self.task.bddl_file)
        self.env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=H,camera_widths=W)
        self.env.reset(); self.sim=self.env.env.sim; m=self.m=self.sim.model
        self.S=np.asarray(torch.load(os.path.join(get_libero_path("init_states"),
            self.task.problem_folder,self.task.init_states_file),weights_only=False))
        self.CID=m.camera_name2id("agentview")
        self.P0,self.Q0=m.cam_pos[self.CID].copy(),m.cam_quat[self.CID].copy()
        self.LP0,self.LD0=m.light_pos.copy(),m.light_dir.copy()
        self.LDF0,self.LSP0=m.light_diffuse.copy(),m.light_specular.copy()
        self.AMB0=np.array(m.vis.headlight.ambient).copy()
        self.CS0=m.light_castshadow.copy()
        self._body_pos0=m.body_pos.copy()
        jn={m.joint_id2name(j) for j in range(m.njnt)}
        # every free object (for collision checks) and obj_of_interest (what gets perturbed)
        self.free_objs=[(m.joint_id2name(j).rsplit("_joint0",1)[0], m.jnt_qposadr[j]+1)
                        for j in range(m.njnt) if m.jnt_type[j]==0]
        self.tgt_objs=[(nm, m.jnt_qposadr[m.joint_name2id(f"{nm}_joint0")]+1)
                       for nm in self.env.obj_of_interest if f"{nm}_joint0" in jn]

        # Look-at point: the point on the optical axis closest to the centroid of the movable
        # objects. Intersecting a fixed z = 0.90 plane instead breaks down, because table
        # height differs per arena and the intersection can land behind the camera -- which
        # flipped the sign of the elevation on libero_object.
        f0=-quat2mat(self.Q0)[:,2]
        self.env.set_init_state(self.S[0]); self.sim.forward()
        cs=[]
        for nm,c in self.free_objs:
            bid=None
            for cand in (f"{nm}_main", nm):
                try: bid=m.body_name2id(cand); break
                except Exception: pass
            if bid is None:
                try: bid=int(m.jnt_bodyid[m.joint_name2id(f"{nm}_joint0")])
                except Exception: pass
            if bid is not None: cs.append(self.sim.data.xipos[bid].copy())
        ctr=np.mean(cs,axis=0) if cs else self.P0+f0
        d0=float(np.dot(ctr-self.P0,f0))
        if not np.isfinite(d0) or d0<0.2: d0=1.1          # fallback for degenerate geometry
        self.L0=self.P0+d0*f0
        u0=self.P0-self.L0; self.D0=np.linalg.norm(u0); u0=u0/self.D0
        self.AZ0,self.EL0=np.arctan2(u0[1],u0[0]),np.arcsin(u0[2])
        self.JR=m.jnt_range[:7].copy()

    def reset_model(self):
        m=self.m
        m.cam_pos[self.CID],m.cam_quat[self.CID]=self.P0,self.Q0
        m.light_pos[:],m.light_dir[:]=self.LP0,self.LD0
        m.light_diffuse[:],m.light_specular[:]=self.LDF0,self.LSP0
        m.vis.headlight.ambient[:]=self.AMB0
        m.light_castshadow[:]=self.CS0
        m.body_pos[:]=self._body_pos0

    def set_camera(self,d):
        az,el,dist,lx,ly=d; m=self.m
        L=self.L0+np.array([lx,ly,0.])
        a,e,dd=self.AZ0+np.radians(az),self.EL0+np.radians(el),self.D0+dist
        p=L+dd*np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
        m.cam_pos[self.CID]=p; m.cam_quat[self.CID]=lookat_quat(p,L)

    def set_lighting(self,d,shadow=False):
        """d = (dEV, d_az_deg, d_el_deg, dlog2T, dlog2_ambient)"""
        ev,az,el,dlt,damb=d; m=self.m
        Rz=rot_z(np.radians(az))
        for i in range(m.nlight):
            p=Rz@(self.LP0[i]-self.L0)+self.L0; dv=Rz@self.LD0[i]
            if abs(el)>1e-12:
                rel=p-self.L0; n=np.linalg.norm(rel[:2])
                ax=np.array([-rel[1],rel[0],0.])/n if n>1e-9 else np.array([0.,1.,0.])
                Re=rot_axis(ax,np.radians(el)); p=Re@(p-self.L0)+self.L0; dv=Re@dv
            m.light_pos[i]=p; m.light_dir[i]=dv/np.linalg.norm(dv)
            m.light_castshadow[i]=1 if shadow else 0
        if shadow: m.vis.quality.shadowsize=max(int(m.vis.quality.shadowsize),4096)
        c=kelvin_rgb(6500.*2**dlt)/kelvin_rgb(6500.); c=c/(0.2126*c[0]+0.7152*c[1]+0.0722*c[2])
        for i in range(m.nlight):
            m.light_diffuse[i]=self.LDF0[i]*(2.**ev)*c; m.light_specular[i]=self.LSP0[i]*(2.**ev)
        m.vis.headlight.ambient[:]=np.clip(self.AMB0*(2.**damb),0,1)

    def set_light5(self, d, shadow=False):
        """The lighting axis, five dimensions.

        d = (d_intensity_ev, d_log2_warm, d_tint_g, d_log2_ambient, d_elevation_deg)

        Shadows are not cast: LIBERO's default castshadow=false is kept. Enabling shadows makes
        MuJoCo stop applying the spotlight's cone falloff (cutoff 45 deg), and in scenes with a
        wide table that overflows the cone (libero_10 tasks 2, 3, 8, 9) exposure jumps by 1.28x
        and half the frame blows out. This is presumably why LIBERO disables them too.

        With no shadows the light's *azimuth* barely affects the image, so it is not part of the
        axis: spending one of five spherical dimensions on a parameter with no effect would
        dilute the others.

        Colour is parameterised as a symmetric logarithmic gain. Following the blackbody locus
        (6500K * 2^d) instead is asymmetric: pushed hard, the warm side saturates to pure red
        (1,0,0) while the cool side barely moves.
          warm : R x2^d,  B x2^-d          (R/B ratio = 2^(2d); 16x at L3)
          tint : G x2^d,  R,B x2^(-d/2)    (green / magenta; G x2.14 at L3)
        Both are normalised to preserve Rec.709 luminance, so only the colour changes.
        """
        ev, warm, tint, damb, el = d
        m = self.m
        for i in range(m.nlight):
            p = self.LP0[i].copy(); dv = self.LD0[i].copy()
            if abs(el) > 1e-12:
                rel = p - self.L0; n = np.linalg.norm(rel[:2])
                ax = np.array([-rel[1], rel[0], 0.]) / n if n > 1e-9 else np.array([0., 1., 0.])
                Re = rot_axis(ax, np.radians(el)); p = Re @ (p - self.L0) + self.L0; dv = Re @ dv
            m.light_pos[i] = p; m.light_dir[i] = dv / np.linalg.norm(dv)
            m.light_castshadow[i] = 1 if shadow else 0
        c = light_color(warm, tint)
        for i in range(m.nlight):
            m.light_diffuse[i] = self.LDF0[i] * (2. ** ev) * c
            m.light_specular[i] = self.LSP0[i] * (2. ** ev)
        m.vis.headlight.ambient[:] = np.clip(self.AMB0 * (2. ** damb), 0, 1)

    def perturb_robot(self,st,dq):
        st=st.copy(); st[1:8]=np.clip(st[1:8]+dq,self.JR[:,0],self.JR[:,1]); return st
    def perturb_objects(self,st,dxy):
        st=st.copy()
        for k,(nm,c) in enumerate(self.tgt_objs):
            st[c]+=dxy[2*k]; st[c+1]+=dxy[2*k+1]
        return st

    def restore(self):
        """Reset the controller's internal state after stepping. env.reset() is avoided here
        because it rebuilds the sim."""
        try: self.env.env.robots[0].controller.reset_goal()
        except Exception: pass

    def shot(self,st,cam="agentview"):
        self.env.set_init_state(st); self.sim.forward()
        return self.sim.render(width=self.W,height=self.H,camera_name=cam)[::-1]
    def close(self): self.env.close()


# ---------------- robot perturbation in end-effector space (IK) ----------------
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

def robot_collides(T, st, tol=0.002):
    """Whether the arm is interpenetrating the table, a fixture or an object."""
    m,sim=T.m,T.sim
    T.env.set_init_state(st); sim.forward()
    worst=0.0; pair=None
    for i in range(sim.data.ncon):
        con=sim.data.contact[i]
        if con.dist>=-tol: continue
        n1=m.body_id2name(m.geom_bodyid[con.geom1]); n2=m.body_id2name(m.geom_bodyid[con.geom2])
        rob=lambda n: n is not None and (n.startswith("robot0") or n.startswith("gripper0"))
        if rob(n1)!=rob(n2):
            if -con.dist>worst: worst=-con.dist; pair=(n1,n2)
    return (worst>tol), worst, pair


# ---------------- detect what an object rests on, and move the support with it ----------------
from collections import defaultdict

def _root_body(T, bid):
    """The ancestor body id, walked up to the child of world."""
    m=T.m
    while m.body_parentid[bid] != 0: bid = m.body_parentid[bid]
    return bid

def _is_robot(name):
    return name is not None and (name.startswith("robot0") or name.startswith("gripper0"))

def detect_supports(T, st, warm=20):
    """Settle the nominal scene and report what rests on what.

    Returns supports {free object -> supporting root body name} and
    resting_on {root body name -> [free objects]}."""
    m,sim=T.m,T.sim
    T.env.set_init_state(st); sim.forward()
    dummy=np.zeros(T.env.env.action_dim); dummy[-1]=-1.0
    for _ in range(warm): T.env.step(dummy)
    # body id -> free object name
    obj_of_body={}
    for nm,_ in T.free_objs:
        try: obj_of_body[m.body_name2id(f"{nm}_main")]=nm
        except Exception: pass
    supports={}; resting=defaultdict(list); best={}
    for i in range(sim.data.ncon):
        con=sim.data.contact[i]
        if con.dist > 1e-3: continue
        for ga,gb in ((con.geom1,con.geom2),(con.geom2,con.geom1)):
            ba,bb=m.geom_bodyid[ga],m.geom_bodyid[gb]
            o=obj_of_body.get(ba)
            if o is None: continue
            if _is_robot(m.body_id2name(bb)): continue
            zc=con.pos[2]; zo=sim.data.xipos[ba][2]
            if zc < zo - 0.005:                       # contact below the centroid = support
                depth=zo-zc
                if o not in best or depth>best[o][0]:
                    best[o]=(depth,_root_body(T,bb))
    for o,(_,rb) in best.items():
        rn=m.body_id2name(rb)
        sup=obj_of_body.get(rb, rn)                   # a free object contributes its own name
        supports[o]=sup; resting[sup].append(o)
    T.restore()
    return supports, dict(resting)

def support_move_set(T, supports, resting, targets):
    """Collect the targets, their supports, and everything else resting on those supports.
    The table is never moved. Returns (set of free object names, set of fixture root body names)."""
    free_names={nm for nm,_ in T.free_objs}
    is_table=lambda n: n is None or "table" in n
    mf=set(targets); mx=set()
    for o in targets:
        cur=o
        for _ in range(6):
            s=supports.get(cur)
            if s is None or is_table(s): break
            (mf if s in free_names else mx).add(s); cur=s
    ch=True
    while ch:
        ch=False
        for b in list(mf|mx):
            for o in resting.get(b,[]):
                if o not in mf: mf.add(o); ch=True
    return mf, mx

def perturb_objects_support(T, st, dxy_map, fixture_shift):
    """Displace free objects through qpos and fixtures through the model's body_pos. Because
    this writes to the model, the caller must call reset_fixtures() afterwards."""
    st=st.copy(); m=T.m
    cols={nm:c for nm,c in T.free_objs}
    for nm,d in dxy_map.items():
        c=cols[nm]; st[c]+=d[0]; st[c+1]+=d[1]
    for rn,d in fixture_shift.items():
        bid=m.body_name2id(rn)
        m.body_pos[bid]=T._body_pos0[bid]+np.array([d[0],d[1],0.0])
    return st

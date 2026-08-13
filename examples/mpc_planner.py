"""
Linear time-varying MPC planner for the F1TENTH gym.

Tracks the same raceline used by the pure-pursuit example
(example_waypoints.csv) with a kinematic-bicycle-model MPC. Each control
step re-linearizes (successive convexification) around a nonlinear
forward simulation of the current state, and solves the resulting convex
QP with cvxpy/OSQP. A "pace car" reference (see calc_ref_trajectory) and
a friction-circle steering bound keep the commands the linear model
proposes within what the real, slip-limited vehicle can actually deliver.

Known limitation: the kinematic model carries no notion of tire slip, so
in a tight or heavily off-nominal recovery (e.g. a large initial heading
error) tracking precision can fall short of what the raceline's designed
speed profile assumes, and the car can run wide enough to leave the
track. See benchmark_laptime.py for a head-to-head against pure pursuit.
"""
import cvxpy
import numpy as np


class MPCPlanner:
    """
    Kinematic-bicycle-model LTV-MPC.

    State:   z = [x, y, v, yaw]
    Input:   u = [accel, steer]
    """

    def __init__(self, conf, wb,
                 horizon=16, dt=0.05,
                 wheelbase=None,
                 max_steer=0.4189, max_dsteer=3.2,
                 max_accel=9.51, max_speed=15.0, min_speed=0.0,
                 speed_margin=1.05, mu=1.0489, lat_accel_margin=0.9):
        self.conf = conf
        self.wheelbase = wheelbase if wheelbase is not None else wb
        self.horizon = horizon
        self.dt = dt
        self.max_steer = max_steer
        self.max_dsteer = max_dsteer
        self.max_accel = max_accel
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_margin = speed_margin
        self.mu = mu
        self.lat_accel_max = lat_accel_margin * mu * 9.81

        # cost weights: state = [x, y, v, yaw], input = [accel, steer]
        self.Q = np.diag([40.0, 40.0, 3.0, 8.0])
        self.Qf = np.diag([40.0, 40.0, 3.0, 8.0])
        self.R = np.diag([0.01, 1.0])
        self.Rd = np.diag([0.01, 5.0])

        self.load_waypoints(conf)
        self.prev_steer = np.zeros(horizon)
        self.prev_accel = np.zeros(horizon)
        self.last_idx = None
        self.last_applied_u = np.zeros(2)

    def load_waypoints(self, conf):
        waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        self.waypoints = waypoints
        self.wpt_xyz = waypoints[:, [conf.wpt_xind, conf.wpt_yind]]
        self.wpt_yaw = waypoints[:, conf.wpt_thind]
        self.wpt_v = waypoints[:, conf.wpt_vind]
        self.wpt_kappa = waypoints[:, conf.wpt_thind + 1]
        # cumulative arc length along the (closed) raceline, for lookahead spacing
        diffs = np.diff(self.wpt_xyz, axis=0, append=self.wpt_xyz[:1])
        seg_len = np.linalg.norm(diffs, axis=1)
        self.wpt_s = np.concatenate(([0.0], np.cumsum(seg_len)[:-1]))
        self.track_len = np.sum(seg_len)

    def _nearest_index(self, position):
        dists = np.linalg.norm(self.wpt_xyz - position, axis=1)
        return int(np.argmin(dists))

    def calc_ref_trajectory(self, state):
        """Build an (N+1)-step reference [x, y, v, yaw] ahead of the car.

        The reference is a "pace car": it advances along the raceline at
        the raceline's own speed profile, purely as a function of elapsed
        time, independent of where the ego car actually is. (Snapping the
        reference to the nearest waypoint to the *ego* position instead --
        the more obvious approach -- lets the reference stall or loop if
        the car ever drifts off-path, since the nearest point to a car
        stuck circling nearby never advances. A pace car can't get stuck:
        it always moves forward.)
        """
        n_wpts = self.wpt_xyz.shape[0]
        if self.last_idx is None:
            # one-time alignment to the raceline at the car's actual start
            i0 = self._nearest_index(state[:2])
            self.pace_s = self.wpt_s[i0]
            self.last_idx = i0

        ref = np.zeros((4, self.horizon + 1))
        i0 = int(np.searchsorted(self.wpt_s, self.pace_s) % n_wpts)
        ref[0, 0] = self.wpt_xyz[i0, 0]
        ref[1, 0] = self.wpt_xyz[i0, 1]
        ref[2, 0] = self.wpt_v[i0]
        ref[3, 0] = self.wpt_yaw[i0]

        # if the ego car is lagging well behind the pace car, ease off the
        # pace car's advance rate (never all the way to a stop -- that's
        # the stall failure mode this design exists to avoid) so a car that
        # fell behind during e.g. an initial heading correction gets a
        # chance to close the gap instead of permanently chasing a target
        # that's always racing away at full raceline speed. Apply the same
        # throttle across the whole horizon, not just the anchor point --
        # otherwise the lookahead the MPC actually optimizes against still
        # races ahead at full speed even though the anchor was slowed down.
        lag = np.linalg.norm(state[:2] - ref[:2, 0])
        pace_factor = np.clip(1.0 - 0.15 * lag, 0.3, 1.0)
        self.pace_s = (self.pace_s + max(self.wpt_v[i0], 1.0) * self.dt * pace_factor) % self.track_len
        self.last_idx = i0

        travel = self.wpt_s[i0]
        idx = i0
        for t in range(1, self.horizon + 1):
            v_local = max(self.wpt_v[idx], 1.0)
            travel = (travel + v_local * self.dt * pace_factor) % self.track_len
            idx = int(np.searchsorted(self.wpt_s, travel) % n_wpts)
            ref[0, t] = self.wpt_xyz[idx, 0]
            ref[1, t] = self.wpt_xyz[idx, 1]
            ref[2, t] = self.wpt_v[idx]
            ref[3, t] = self.wpt_yaw[idx]

        # unwrap yaw reference relative to current heading so the MPC
        # doesn't see a spurious +-2pi jump
        ref[3, :] = np.unwrap(np.concatenate(([state[3]], ref[3, :])))[1:]
        return ref

    def _step_nonlinear(self, z, a, delta):
        x, y, v, yaw = z
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        x = x + v * np.cos(yaw) * self.dt
        y = y + v * np.sin(yaw) * self.dt
        yaw = yaw + v / self.wheelbase * np.tan(delta) * self.dt
        v = np.clip(v + a * self.dt, self.min_speed, self.max_speed)
        return np.array([x, y, v, yaw])

    def predict_motion(self, z0, accel_seq, steer_seq):
        """Forward-simulate the nonlinear model from the true current state
        using a candidate control sequence, to get an operating-point
        trajectory to linearize around (starts exactly at z0, unlike the
        raceline reference which can be far from the car's actual pose)."""
        z_bar = np.zeros((4, self.horizon + 1))
        z_bar[:, 0] = z0
        z = z0.copy()
        for t in range(self.horizon):
            z = self._step_nonlinear(z, accel_seq[t], steer_seq[t])
            z_bar[:, t + 1] = z
        return z_bar

    def get_linear_model_matrices(self, z_bar, u_bar):
        """First-order Taylor expansion of the nonlinear kinematic model
        around operating point (z_bar, u_bar). Returns A, B, C such that
        z_{t+1} ~= A @ z_t + B @ u_t + C is exact at (z_bar, u_bar) and a
        local approximation nearby."""
        dt, L = self.dt, self.wheelbase
        v_bar, yaw_bar = z_bar[2], z_bar[3]
        delta_bar = u_bar[1]
        cos_yaw, sin_yaw = np.cos(yaw_bar), np.sin(yaw_bar)
        cos_delta = np.cos(delta_bar)

        A = np.array([
            [1.0, 0.0, cos_yaw * dt, -v_bar * sin_yaw * dt],
            [0.0, 1.0, sin_yaw * dt, v_bar * cos_yaw * dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, np.tan(delta_bar) * dt / L, 1.0],
        ])
        B = np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [dt, 0.0],
            [0.0, v_bar * dt / (L * cos_delta ** 2)],
        ])
        # C must use the true nonlinear next state at the operating point,
        # not z_bar itself (which would implicitly assume f(z_bar,u_bar) ==
        # z_bar, i.e. zero dynamics) -- otherwise the affine model drifts
        # away from the real dynamics every step.
        f_bar = self._step_nonlinear(z_bar, u_bar[0], u_bar[1])
        C = f_bar - A @ z_bar - B @ u_bar
        return A, B, C

    def _solve_qp(self, state, ref_traj, z_bar, u_bar_a, u_bar_d):
        """Solve one convex QP linearized around the given operating-point
        trajectory (z_bar/u_bar), tracking ref_traj in cost."""
        N = self.horizon
        z = cvxpy.Variable((4, N + 1))
        u = cvxpy.Variable((2, N))

        cost = 0.0
        constraints = [z[:, 0] == state]

        for t in range(N):
            v_lin = max(z_bar[2, t], 0.5)
            z_lin = np.array([z_bar[0, t], z_bar[1, t], v_lin, z_bar[3, t]])
            u_lin = np.array([u_bar_a[t], u_bar_d[t]])
            A, B, C = self.get_linear_model_matrices(z_lin, u_lin)

            constraints += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
            cost += cvxpy.quad_form(z[:, t] - ref_traj[:, t], self.Q)
            cost += cvxpy.quad_form(u[:, t], self.R)
            # penalize/limit the change from the PREVIOUS CONTROL STEP's
            # actually-applied input at t=0, not just smoothness within this
            # horizon -- otherwise nothing stops consecutive receding-horizon
            # solves (each a fresh QP) from flipping the first action between
            # calls, which was making the steering chatter hard-over to
            # hard-over every 0.1s
            prev_u = self.last_applied_u if t == 0 else u[:, t - 1]
            cost += cvxpy.quad_form(u[:, t] - prev_u, self.Rd)
            if t == 0:
                constraints += [cvxpy.abs(u[1, 0] - self.last_applied_u[1]) <= self.max_dsteer * self.dt]

            constraints += [u[0, t] <= self.max_accel, u[0, t] >= -self.max_accel]
            # friction-circle steer bound: kinematically, steering at delta
            # while moving at v_lin implies lateral accel ~= v_lin^2 * tan(delta) / L.
            # cap delta so that implied lateral accel stays within available
            # tire grip -- without this the kinematic model happily commands
            # steer angles the real (slip-limited) tires can't deliver, which
            # is exactly what was spinning the car out.
            grip_steer_bound = np.arctan(self.lat_accel_max * self.wheelbase / max(v_lin, 0.5) ** 2)
            steer_bound = min(self.max_steer, grip_steer_bound)
            constraints += [u[1, t] <= steer_bound, u[1, t] >= -steer_bound]
            # cap speed to the (already grip-limited) raceline profile plus a
            # small margin -- the kinematic model has no notion of tire slip,
            # so nothing else stops it from commanding an unsafe speed
            speed_cap = min(self.max_speed, ref_traj[2, t] * self.speed_margin)
            constraints += [z[2, t] <= speed_cap, z[2, t] >= self.min_speed]
            if t < N - 1:
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.max_dsteer * self.dt]

        terminal_speed_cap = min(self.max_speed, ref_traj[2, N] * self.speed_margin)
        constraints += [z[2, N] <= terminal_speed_cap, z[2, N] >= self.min_speed]
        cost += cvxpy.quad_form(z[:, N] - ref_traj[:, N], self.Qf)

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, warm_start=True, verbose=False)

        if u.value is None:
            return None, None
        accel = np.asarray(u.value[0, :]).flatten()
        steer = np.asarray(u.value[1, :]).flatten()
        return accel, steer

    def solve(self, state, ref_traj, iterations=3):
        """state = [x, y, v, yaw]. Iteratively relinearizes the kinematic
        model around a nonlinear-forward-simulated operating trajectory
        that starts at the true current state (successive convexification),
        then returns the optimal accel/steer sequences."""
        accel_seq, steer_seq = self.prev_accel.copy(), self.prev_steer.copy()

        for _ in range(iterations):
            z_bar = self.predict_motion(state, accel_seq, steer_seq)
            new_accel, new_steer = self._solve_qp(state, ref_traj, z_bar, accel_seq, steer_seq)
            if new_accel is None:
                break
            accel_seq, steer_seq = new_accel, new_steer

        self.prev_accel, self.prev_steer = accel_seq, steer_seq
        self.last_applied_u = np.array([accel_seq[0], steer_seq[0]])
        return accel_seq, steer_seq

    def _ego_grip_speed_cap(self, position):
        """Grip-limited speed cap based on curvature at the ego car's own
        nearest point on the raceline -- independent of the pace-car index,
        which reflects progress along the reference path, not necessarily
        where the ego car physically is (e.g. if it cut a corner short)."""
        idx = self._nearest_index(position)
        kappa = abs(self.wpt_kappa[idx])
        if kappa < 1e-3:
            return self.max_speed
        return np.sqrt(self.lat_accel_max / kappa)

    def plan(self, pose_x, pose_y, pose_theta, velocity):
        state = np.array([pose_x, pose_y, velocity, pose_theta])
        ref_traj = self.calc_ref_trajectory(state)

        safe_cap = self._ego_grip_speed_cap(state[:2]) * 1.05
        ref_traj[2, :] = np.minimum(ref_traj[2, :], safe_cap)

        accel, steer = self.solve(state, ref_traj)

        target_speed = float(np.clip(velocity + accel[0] * self.dt, self.min_speed, self.max_speed))
        target_steer = float(np.clip(steer[0], -self.max_steer, self.max_steer))
        return target_speed, target_steer

    def render_waypoints(self, *args, **kwargs):
        pass

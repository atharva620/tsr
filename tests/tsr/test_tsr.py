import numpy as np
import math
from tsr.generic import cylinder_grasp, box_grasp

try:
    from tsr.generic import cylinder_grasp, box_grasp 
except ImportError:
    print("Import failed. Adjust the import path to your tsr module.")

class DummyRobot:
    pass


class DummyBox:
    """Just returns a fixed 4x4 transform."""
    def __init__(self, T):
        self._T = T
    def GetTransform(self):
        return self._T


def _get_single_tsr(chain):
    """Some TSRChain impls store .TSRs list; others single attr."""
    if hasattr(chain, "TSRs"):
        return chain.TSRs[0]
    if hasattr(chain, "TSR"):
        return chain.TSR
    raise AttributeError("Cannot find TSR(s) on chain object")


def _print_chains(label, chains):
    print(f"\n=== {label} ({len(chains)} chains) ===")
    for i, c in enumerate(chains):
        t = _get_single_tsr(c)
        print(f"[{i}] Tw_e:\n{t.Tw_e}")
        print(f"    Bw:\n{t.Bw}")
        print(f"    manipindex: {t.manipindex}")


def test_cylinder_grasp_basic():
    robot = DummyRobot()
    obj_pos = np.array([1.0, 2.0, 3.0])
    r = 0.10
    h = 0.40
    lateral = 0.02
    vt = 0.05
    yaw_rng = [-math.pi/2, math.pi/2]
    chains = cylinder_grasp(robot, obj_pos, r, h,
                            lateral_offset=lateral,
                            vertical_tolerance=vt,
                            yaw_range=yaw_rng,
                            manip_idx=0)

    assert len(chains) == 2

    total_off = lateral + r
    z_center = 0.5 * h

    # Check chain 0
    tsr0 = _get_single_tsr(chains[0])
    np.testing.assert_allclose(tsr0.T0_w[:3, 3], obj_pos, atol=1e-9)
    np.testing.assert_allclose(tsr0.Tw_e[0, 3], -total_off, atol=1e-9)
    np.testing.assert_allclose(tsr0.Tw_e[2, 3], z_center, atol=1e-9)
    np.testing.assert_allclose(tsr0.Bw[2], [-vt, vt], atol=1e-9)
    np.testing.assert_allclose(tsr0.Bw[5], yaw_rng, atol=1e-9)

    # Check chain 1 (flipped hand); same translations
    tsr1 = _get_single_tsr(chains[1])
    np.testing.assert_allclose(tsr1.Tw_e[0, 3], -total_off, atol=1e-9)
    np.testing.assert_allclose(tsr1.Tw_e[2, 3], z_center, atol=1e-9)
    np.testing.assert_allclose(tsr1.Bw[2], [-vt, vt], atol=1e-9)
    np.testing.assert_allclose(tsr1.Bw[5], yaw_rng, atol=1e-9)


def test_cylinder_grasp_input_validation():
    robot = DummyRobot()
    p = np.zeros(3)
    try:
        cylinder_grasp(robot, p, -0.1, 0.2)
        assert False, "negative radius should raise"
    except Exception:
        pass
    try:
        cylinder_grasp(robot, p, 0.1, -0.2)
        assert False, "negative height should raise"
    except Exception:
        pass
    try:
        cylinder_grasp(robot, p, 0.1, 0.2, vertical_tolerance=-0.01)
        assert False, "negative vertical_tolerance should raise"
    except Exception:
        pass
    try:
        cylinder_grasp(robot, p, 0.1, 0.2, yaw_range=[0])
        assert False, "bad yaw_range length should raise"
    except Exception:
        pass
    try:
        cylinder_grasp(robot, p, 0.1, 0.2, yaw_range=[1.0, -1.0])
        assert False, "min>max yaw_range should raise"
    except Exception:
        pass


def test_box_grasp_basic():
    # Box pose @ (0.5, -0.2, 0.1), identity orientation
    T = np.eye(4)
    T[:3, 3] = [0.5, -0.2, 0.1]
    box = DummyBox(T)
    L, W, H = 0.4, 0.2, 0.6
    lateral = 0.03
    tol = 0.01
    chains = box_grasp(DummyRobot(), box, L, W, H,
                       manip_idx=1,
                       lateral_offset=lateral,
                       lateral_tolerance=tol)

    # Expect 12 chains (6 faces × 2 orientations)
    assert len(chains) == 12

    # Grab top face chain 0: translation z = lateral + height
    tsr0 = _get_single_tsr(chains[0])
    np.testing.assert_allclose(tsr0.T0_w[:3, 3], [0.5, -0.2, 0.1], atol=1e-9)
    np.testing.assert_allclose(tsr0.Tw_e[2, 3], lateral + H, atol=1e-9)
    np.testing.assert_allclose(tsr0.Bw[1], [-tol, tol], atol=1e-9)

    # Bottom face (chain 1): translation z = -lateral_offset
    tsr1 = _get_single_tsr(chains[1])
    np.testing.assert_allclose(tsr1.Tw_e[2, 3], -lateral, atol=1e-9)
    np.testing.assert_allclose(tsr1.Bw[1], [-tol, tol], atol=1e-9)

    # Front face (chain 2): translation x = +0.5*L + lateral
    tsr2 = _get_single_tsr(chains[2])
    np.testing.assert_allclose(tsr2.Tw_e[0, 3], 0.5*L + lateral, atol=1e-9)
    np.testing.assert_allclose(tsr2.Tw_e[2, 3], 0.5*H, atol=1e-9)

    # Side face (chain 4): translation y = +0.5*W + lateral
    tsr4 = _get_single_tsr(chains[4])
    np.testing.assert_allclose(tsr4.Tw_e[1, 3], 0.5*W + lateral, atol=1e-9)

    # Rotated copy (chain 6): should match chain 0 but rotated 180° about z.
    tsr6 = _get_single_tsr(chains[6])
    # z trans should stay same as top
    np.testing.assert_allclose(tsr6.Tw_e[2, 3], lateral + H, atol=1e-9)


def test_box_grasp_input_validation():
    box = DummyBox(np.eye(4))
    try:
        box_grasp(DummyRobot(), box, -0.1, 0.2, 0.3, manip_idx=0)
        assert False, "negative length should raise"
    except Exception:
        pass
    try:
        box_grasp(DummyRobot(), box, 0.1, -0.2, 0.3, manip_idx=0)
        assert False, "negative width should raise"
    except Exception:
        pass
    try:
        box_grasp(DummyRobot(), box, 0.1, 0.2, -0.3, manip_idx=0)
        assert False, "negative height should raise"
    except Exception:
        pass
    try:
        box_grasp(DummyRobot(), box, 0.1, 0.2, 0.3, manip_idx=0,
                  lateral_tolerance=-0.01)
        assert False, "negative lateral_tolerance should raise"
    except Exception:
        pass

if __name__ == "__main__":
    # Run the tests (simple manual invocation)
    test_cylinder_grasp_basic()
    test_cylinder_grasp_input_validation()
    test_box_grasp_basic()
    test_box_grasp_input_validation()

    # Re‑run w/ printouts so you can inspect
    print("\n\n--- Manual inspection run ---")

    robot = DummyRobot()
    obj_pos = np.array([1.0, 2.0, 3.0])
    cyl_chains = cylinder_grasp(robot, obj_pos,
                                obj_radius=0.10,
                                obj_height=0.40,
                                lateral_offset=0.02,
                                vertical_tolerance=0.05,
                                yaw_range=[-math.pi/2, math.pi/2],
                                manip_idx=0)
    _print_chains("Cylinder grasp", cyl_chains)

    T = np.eye(4); T[:3,3] = [0.5, -0.2, 0.1]
    box = DummyBox(T)
    box_chains = box_grasp(robot, box,
                           length=0.4, width=0.2, height=0.6,
                           manip_idx=1,
                           lateral_offset=0.03,
                           lateral_tolerance=0.01)
    _print_chains("Box grasp", box_chains)

    print("\nAll assertions passed.")

"""Independent toy scenes expose overlapping screening and retained-stem effects."""

import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch
from leaf_virtual_experiments import screening_bookkeeping, compose_owners, layered_raster, EPS


class VirtualExperimentTests(unittest.TestCase):
    def test_accepted_family_preserves_fresh_subset_and_requires_baseline(self):
        import leaf_virtual_experiments as module

        validate = getattr(module, "validate_candidate_ids", None)
        self.assertIsNotNone(validate)
        fresh = [0, 1, 3, 4, 6, 10, 14]
        self.assertEqual(validate(np.array(fresh), 7), fresh)
        self.assertEqual(validate(np.array([0, 4, 19]), 3), [0, 4, 19])
        for ids, count in [([1, 3], 2), ([0, 0], 2), ([0, 1], 3), ([], 0)]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate(np.array(ids), count)

    def test_convergence_reports_actual_seven_configuration_family(self):
        import itertools
        import json
        import leaf_virtual_experiments as module

        ids = [0, 1, 3, 4, 6, 10, 14]
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            for name in ["experiments", "experiments_512"]:
                folder = dest / name
                folder.mkdir()
                edges = [
                    dict(
                        candidate_id=cid,
                        direction=f"beam_{direction}",
                        removed_leaf=f"leaf_{donor}",
                        receiver_leaf=f"leaf_{receiver}",
                        rescue_fraction=0.1,
                        isolated_pixels=100,
                        pixel_area_H2=0.001,
                    )
                    for cid in ids
                    for direction in range(9)
                    for donor, receiver in itertools.permutations(range(1, 6), 2)
                ]
                states = [
                    dict(
                        candidate_id=cid,
                        direction=f"beam_{direction}",
                        receiver_leaf=f"leaf_{receiver}",
                        redundant_fraction=0.2,
                        near_tie_fraction=0,
                        isolated_pixels=100,
                    )
                    for cid in ids
                    for direction in range(9)
                    for receiver in range(1, 6)
                ]
                module.write_csv(folder / "directed_neighbour_effects.csv", edges)
                module.write_csv(folder / "screening_decomposition.csv", states)
                (folder / "summary.json").write_text("{}", encoding="utf-8")
            # Materialize the producer inside the fixture workspace, as the
            # real runner does. Hosted Windows source and Temp can use different drives.
            producer = dest / "leaf_virtual_experiments.py"
            producer.write_bytes(Path(module.__file__).read_bytes())
            with (
                patch.object(module, "DEST", dest),
                patch.object(module, "ROOT", dest),
                patch.object(module, "__file__", str(producer)),
            ):
                module.convergence()
            result = json.loads(
                (dest / "numerical_stability_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result.get("configuration_count"), 7)
            self.assertEqual(result.get("candidate_ids"), ids)
            self.assertEqual(result.get("resolved_edges_all_configurations"), 20)
            self.assertTrue(
                all(e.get("all_configurations_resolved_positive") for e in result["edge_summaries"])
            )

    def test_two_redundant_blockers_require_joint_removal(self):
        # Receiver0 fills four pixels. Leaves1/2 both cover the same first two.
        d = np.full((4, 1, 4), -np.inf)
        d[0] = 0
        d[1, 0, :2] = 1
        d[2, 0, :2] = 2
        r = screening_bookkeeping(d, 0, [0, 1, 2], 3)
        self.assertEqual(r["footprint_pixels"], 4)
        self.assertEqual(r["full_pixels"], 2)
        self.assertEqual(r["exclusive_pixels"], 0)
        self.assertEqual(r["redundant_pixels"], 2)
        self.assertEqual([x["received_pixels"] for x in r["coalitions"]], [2, 2, 2, 4])
        self.assertEqual(sum(r["multiplicity_pixels"]), 4)

    def test_stem_stays_present_after_every_neighbour_is_removed(self):
        d = np.full((4, 1, 4), -np.inf)
        d[0] = 0
        d[1] = 1
        d[3, 0, 0] = 2
        r = screening_bookkeeping(d, 0, [0, 1, 2], 3)
        self.assertEqual(r["stem_blocked_pixels"], 1)
        self.assertEqual(r["all_neighbours_removed_pixels"], 3)
        self.assertEqual(r["exclusive_pixels"], 3)
        self.assertEqual(r["redundant_pixels"], 0)

    def test_ties_keep_earlier_organ_and_effects_are_directed(self):
        d = np.array([[[1.0, 1.0]], [[1.0, 0.0]], [[0.0, -np.inf]]])
        np.testing.assert_array_equal(compose_owners(d, [0, 1, 2]), [[0, 0]])
        a = screening_bookkeeping(d, 1, [0, 1], 2)
        b = screening_bookkeeping(d, 0, [0, 1], 2)
        self.assertEqual(a["single_removal_pixels"][0], 2)
        self.assertEqual(b["single_removal_pixels"][1], 0)

    def test_all_subset_outcomes_are_monotone(self):
        rng = np.random.default_rng(24)
        d = rng.normal(size=(6, 4, 5))
        d[0] = 0
        r = screening_bookkeeping(d, 0, list(range(5)), 5)
        values = {x["removal_bits"]: x["received_pixels"] for x in r["coalitions"]}
        for a, va in values.items():
            for b, vb in values.items():
                if a & b == a:
                    self.assertLessEqual(va, vb)
        self.assertEqual(
            r["footprint_pixels"],
            r["full_pixels"]
            + r["stem_blocked_pixels"]
            + r["exclusive_pixels"]
            + r["redundant_pixels"],
        )

    def test_near_tie_chain_has_one_global_nearest_depth_order(self):
        depths = np.array([[[0.0]], [[0.6 * EPS]], [[1.2 * EPS]], [[-np.inf]]])
        np.testing.assert_array_equal(compose_owners(depths, [2, 0, 1, 3]), [[2]])
        for receiver in range(3):
            result = screening_bookkeeping(depths, receiver, [0, 1, 2], 3)
            self.assertEqual(result["full_pixels"], int(receiver == 2))
            for outcome in result["coalitions"]:
                active = [i for i in range(4) if i not in outcome["removed_ids"]]
                direct = int(np.count_nonzero(compose_owners(depths, active) == receiver))
                self.assertEqual(outcome["received_pixels"], direct)

    def test_positive_sub_epsilon_depth_wins_and_exact_tie_keeps_lower_id(self):
        depths = np.array([[[1.0, 1.0]], [[1.0 + 0.25 * EPS, 1.0]]])
        np.testing.assert_array_equal(compose_owners(depths, [1, 0]), [[1, 0]])

    def test_direct_raster_and_layers_share_strict_depth_and_exact_tie_rule(self):
        from additive_geometry import directional_raster

        sheet = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        for delta in [0.0, 0.25 * EPS]:
            vertices = np.concatenate([sheet, sheet + [0.0, 0.0, delta]])
            organs = np.array([0, 0, 1, 1])
            # Reverse traversal to ensure exact ties use ID rather than face order.
            direct = directional_raster(vertices, faces[::-1], organs[::-1], [0, 0, 1], 32)
            layers, _ = layered_raster(vertices, faces[::-1], organs[::-1], [0, 0, 1], 32)
            expected = compose_owners(layers, [0, 1])
            np.testing.assert_array_equal(direct["owner"], expected)
            self.assertTrue((expected[expected >= 0] == int(delta > 0)).all())

    def test_layered_raster_keeps_true_nearest_depth_within_one_organ(self):
        sheet = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
        vertices = np.concatenate([sheet, sheet + [0.0, 0.0, 0.25 * EPS]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        layers, _ = layered_raster(vertices, faces, np.zeros(4, int), [0, 0, 1], 32)
        self.assertTrue(np.all(layers[0][np.isfinite(layers[0])] > 0.0))


if __name__ == "__main__":
    unittest.main()

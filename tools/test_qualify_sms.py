"""CPU contract tests for the SMS qualification harness; no GPU work."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools import qualify_sms as q


class QualificationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_out = q.OUT
        q.OUT = Path(self.temp.name)
        self.scene = Path("scene.toml")
        self.prov = patch.object(q, "provenance", return_value={"test": "state"})
        self.prov.start()

    def tearDown(self):
        self.prov.stop()
        q.OUT = self.old_out
        self.temp.cleanup()

    def test_read_checked_rejects_nonfinite(self):
        image = q.OUT / "image.exr"
        image.touch()
        with (patch.object(q, "load_exr", return_value=np.full((2, 2, 3), np.nan)),
                  self.assertRaisesRegex(SystemExit, "Nonfinite pixel")):
                q.read_checked(image)
        with patch.object(q, "load_exr", return_value=np.ones((2, 2, 3))):
            self.assertEqual(q.read_checked(image).shape, (2, 2, 3))

    def test_roi_mse_rejects_bad_shapes_and_nan(self):
        reference, candidate = q.OUT / "ref.exr", q.OUT / "cand.exr"
        reference.touch()
        candidate.touch()
        with (patch.object(q, "load_exr", side_effect=[np.ones((240, 320, 3)), np.ones((2, 2, 3))]),
                  self.assertRaisesRegex(SystemExit, "Mismatched image dimensions")):
                q.roi_mse(reference, candidate, q.CASES["C1"]["roi"])
        with (patch.object(q, "load_exr", side_effect=[np.ones((240, 320, 3)), np.full((240, 320, 3), np.nan)]),
                  self.assertRaisesRegex(SystemExit, "Nonfinite pixel")):
                q.roi_mse(reference, candidate, q.CASES["C1"]["roi"])

    def test_roi_and_reference_validation(self):
        path = q.OUT / "image.exr"
        path.touch()
        with patch.object(q, "load_exr", return_value=np.ones((240, 320, 3))):
            with self.assertRaisesRegex(SystemExit, "Invalid normalized ROI"):
                q.roi_energy(path, [0.5, 0.0, 0.5, 1.0])
            self.assertEqual(q.reference_energy(path, q.CASES["C1"]["roi"]).tolist(), [2541.0] * 3)
        with (patch.object(q, "load_exr", return_value=np.zeros((240, 320, 3))),
                  self.assertRaisesRegex(SystemExit, "unusable RGB energy")):
                q.reference_energy(path, q.CASES["C1"]["roi"])

    def test_frame_seconds_rejects_missing_or_duplicate_records(self):
        log = q.OUT / "timing.log"
        log.write_text("frame_timing_ms:0:1.0\nframe_timing_ms:1:2.0\n", encoding="utf-8")
        self.assertAlmostEqual(q.frame_seconds(log, 2), 0.003)
        log.write_text("frame_timing_ms:0:1.0\nframe_timing_ms:0:2.0\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "incomplete per-frame timing"):
            q.frame_seconds(log, 2)

    def test_stale_reference_provenance_is_rejected(self):
        reference = q.OUT / "ref.exr"
        reference.write_bytes(b"reference")
        (q.OUT / "run_manifest.json").write_text(json.dumps([{
            "stage": "reference", "case": "C1", "status": "accepted", "file": reference.name,
            "reference_sha256": q.sha256(reference), "provenance": {"test": "old"},
        }]), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "provenance is stale"):
            q.load_reference("C1")

    def test_stale_output_is_removed_before_render(self):
        output = q.OUT / "stale.exr"
        output.write_bytes(b"stale")
        output.with_suffix(".png").write_bytes(b"stale")

        def fake_run(label, exe, args):
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".png").exists())
            output.touch()
            output.with_suffix(".png").touch()
            log = q.OUT / "stale.log"
            log.write_text("frame_timing_ms:0:1.0\n", encoding="utf-8")
            return 0, log, 100.0

        with patch.object(q, "run_exe", side_effect=fake_run), patch.object(q, "read_checked", return_value=np.ones((1, 1, 3))):
            _, elapsed = q.render_theia("stale", self.scene, "local", 1, 17)
        self.assertAlmostEqual(elapsed, 0.001)

    def test_correctness_pools_both_modes_and_rejects_nan(self):
        record = []
        with (patch.object(q, "prepare_run_dir", return_value=self.scene),
              patch.object(q, "render_theia", side_effect=lambda label, scene, mode, frames, seed: (Path(label + ".exr"), 1.0)),
              patch.object(q, "roi_energy", return_value=np.array([10.0, 20.0, 30.0])),
              patch.object(q, "record", side_effect=record.append),
              patch.object(q, "roi_mse", side_effect=lambda ref, cand, roi: (
                  (2.0 if "local" in cand.name else 1.0) if "512" in cand.name else
                  (1.0 if "local" in cand.name else 0.5)))):
                q.stage_correctness("C1", Path("reference.exr"))
        self.assertEqual(record[0]["results"]["local_512"]["mse"], 2.0)
        self.assertEqual(record[0]["results"]["reuse_512"]["mse"], 1.0)
        with (patch.object(q, "prepare_run_dir", return_value=self.scene),
              patch.object(q, "render_theia", side_effect=lambda label, scene, mode, frames, seed: (Path(label + ".exr"), 1.0)),
              patch.object(q, "roi_energy", return_value=np.array([10.0, 20.0, 30.0])),
              patch.object(q, "roi_mse", return_value=float("nan")),
              self.assertRaisesRegex(SystemExit, "nonfinite metric")):
                q.stage_correctness("C1", Path("reference.exr"))

    def test_performance_executes_all_modes_and_pools_seeds(self):
        record = []

        def render(label, scene, mode, frames, seed):
            return Path(label + ".exr"), frames / 32.0

        with (patch.object(q, "prepare_run_dir", return_value=self.scene),
              patch.object(q, "render_theia", side_effect=render),
              patch.object(q, "roi_energy", return_value=np.array([10.0, 20.0, 30.0])),
              patch.object(q, "roi_mse", side_effect=lambda ref, cand, roi: 0.5 if "reuse" in cand.name else 1.0),
              patch.object(q, "record", side_effect=record.append)):
            q.stage_performance("C1", Path("reference.exr"))
        self.assertEqual(record[0]["status"], "accepted")
        self.assertEqual(record[0]["results"]["reuse"]["2.0"]["frames"], 64)
        self.assertEqual(record[0]["results"]["reuse"]["2.0"]["mse"], 0.5)
        self.assertEqual(set(record[0]["results"]), {"off", "local", "reuse"})

    def test_performance_refinement_is_timing_only(self):
        record = []

        def render(label, scene, mode, frames, seed):
            if label == f"timecheck_C1_reuse_2.0_{frames}":
                return Path(label + ".exr"), 3.0
            if label.startswith(("timecheck2_", "eqt_")) and mode == "reuse" and "2.0" in label:
                return Path(label + ".exr"), 2.0
            return Path(label + ".exr"), frames / 32.0

        with (patch.object(q, "prepare_run_dir", return_value=self.scene),
              patch.object(q, "render_theia", side_effect=render),
              patch.object(q, "roi_energy", return_value=np.array([10.0, 20.0, 30.0])),
              patch.object(q, "roi_mse", side_effect=lambda ref, cand, roi: 0.5 if "reuse" in cand.name else 1.0),
              patch.object(q, "record", side_effect=record.append)):
            q.stage_performance("C1", Path("reference.exr"))
        self.assertEqual(record[0]["results"]["reuse"]["2.0"]["frames"], 43)
        self.assertTrue(record[0]["results"]["reuse"]["2.0"]["refined"])


    def test_performance_budget_deviation_is_inconclusive(self):
        record = []

        def render(label, scene, mode, frames, seed):
            if mode == "off" and seed == 2026 and "8.0" in label and label.startswith("eqt_"):
                return Path(label + ".exr"), 10.0
            return Path(label + ".exr"), frames / 32.0

        with (patch.object(q, "prepare_run_dir", return_value=self.scene),
              patch.object(q, "render_theia", side_effect=render),
              patch.object(q, "roi_energy", return_value=np.array([10.0, 20.0, 30.0])),
              patch.object(q, "roi_mse", side_effect=lambda ref, cand, roi: 0.5 if "reuse" in cand.name else 1.0),
              patch.object(q, "record", side_effect=record.append),
              self.assertRaisesRegex(SystemExit, "timing inconclusive")):
                q.stage_performance("C1", Path("reference.exr"))
        self.assertEqual(record[0]["status"], "timing-inconclusive")
        self.assertFalse(record[0]["results"]["off"]["8.0"]["timing_within_5_percent"])



    def test_shared_sms_sphere_casters_have_outward_winding(self):
        for path in sorted(q.ASSETS.glob("sms_sphere_*.obj")):
            vertices, faces = [], []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("v "):
                    vertices.append(tuple(map(float, line.split()[1:4])))
                elif line.startswith("f "):
                    faces.append(tuple(int(value.split("/")[0]) - 1 for value in line.split()[1:4]))
            points = np.array(vertices)
            triangles = points[np.array(faces)]
            geometric = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            outward = (geometric * triangles.mean(axis=1)).sum(axis=1)
            self.assertGreater(float(outward.min()), 0.0, f"Inward caster winding: {path.name}")



if __name__ == "__main__":
    unittest.main()

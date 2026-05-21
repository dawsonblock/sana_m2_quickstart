import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from sana_core.schemas import GenerationResult
import sana_core.gallery as gallery
import sana_core.presets as presets


class ApiServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_engine_module = sys.modules.get("sana_core.engine")

        fake_engine = types.ModuleType("sana_core.engine")

        def fake_generate_image(request, progress=None):
            metadata = {
                "prompt": request.prompt,
                "seed": request.seed,
                "image_path": request.output or "outputs/generated.png",
                "metadata_path": "outputs/generated.json",
            }
            return GenerationResult(
                image_path=metadata["image_path"],
                metadata_path=metadata["metadata_path"],
                metadata=metadata,
            )

        fake_engine.generate_image = fake_generate_image
        fake_engine.get_device = lambda: "cpu"
        sys.modules["sana_core.engine"] = fake_engine

        cls.api_server = importlib.import_module("api_server")

        from fastapi.testclient import TestClient

        cls.client = TestClient(cls.api_server.app)

    @classmethod
    def tearDownClass(cls):
        if cls.original_engine_module is None:
            sys.modules.pop("sana_core.engine", None)
        else:
            sys.modules["sana_core.engine"] = cls.original_engine_module
        sys.modules.pop("api_server", None)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.outputs = self.root / "outputs"
        self.static = self.root / "static"
        self.presets_dir = self.root / "presets"
        self.logs = self.root / "logs"

        self.outputs.mkdir(parents=True, exist_ok=True)
        self.static.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

        (self.static / "gallery.html").write_text(
            "<html><body>gallery</body></html>",
            encoding="utf-8",
        )
        (self.outputs / "sample.png").write_bytes(b"fake-png")
        (self.outputs / "sample.json").write_text(
            json.dumps(
                {
                    "prompt": "sample prompt",
                    "seed": 7,
                    "image_path": "outputs/sample.png",
                }
            ),
            encoding="utf-8",
        )
        (self.presets_dir / "prompt_presets.json").write_text(
            json.dumps({"presets": []}),
            encoding="utf-8",
        )

        self.original_api_root = self.api_server.PROJECT_ROOT
        self.original_api_log_dir = self.api_server.LOG_DIR
        self.original_gallery_root = gallery.PROJECT_ROOT
        self.original_gallery_output = gallery.OUTPUT_DIR
        self.original_preset_dir = presets.PRESET_DIR
        self.original_preset_path = presets.PRESET_PATH

        self.api_server.PROJECT_ROOT = self.root
        self.api_server.LOG_DIR = self.logs
        gallery.PROJECT_ROOT = self.root
        gallery.OUTPUT_DIR = self.outputs
        presets.PRESET_DIR = self.presets_dir
        presets.PRESET_PATH = self.presets_dir / "prompt_presets.json"

    def tearDown(self):
        self.api_server.PROJECT_ROOT = self.original_api_root
        self.api_server.LOG_DIR = self.original_api_log_dir
        gallery.PROJECT_ROOT = self.original_gallery_root
        gallery.OUTPUT_DIR = self.original_gallery_output
        presets.PRESET_DIR = self.original_preset_dir
        presets.PRESET_PATH = self.original_preset_path
        self.temp_dir.cleanup()

    def test_health_and_models(self):
        health_response = self.client.get("/health")
        models_response = self.client.get("/models")

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json()["device"], "cpu")
        self.assertEqual(models_response.status_code, 200)
        self.assertTrue(models_response.json()["models"])

    def test_generate_uses_stubbed_engine(self):
        response = self.client.post(
            "/generate",
            json={"prompt": "test prompt", "seed": 99},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["metadata"]["prompt"], "test prompt")
        self.assertEqual(payload["metadata"]["seed"], 99)

    def test_presets_metadata_and_gallery_routes(self):
        save_response = self.client.post(
            "/presets",
            json={
                "id": "demo",
                "name": "Demo",
                "prompt": "clean studio light",
            },
        )
        presets_response = self.client.get("/presets")
        metadata_response = self.client.get("/metadata")
        gallery_response = self.client.get("/gallery")
        output_response = self.client.get("/outputs/sample.png")
        missing_file_route_response = self.client.get("/file/outputs/sample.png")

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(presets_response.status_code, 200)
        self.assertEqual(len(presets_response.json()["items"]), 1)
        self.assertEqual(metadata_response.status_code, 200)
        self.assertIn("outputs/sample.json", metadata_response.json()["items"])
        self.assertEqual(gallery_response.status_code, 200)
        self.assertEqual(output_response.status_code, 200)
        self.assertEqual(missing_file_route_response.status_code, 404)

    def test_metadata_endpoint_rejects_non_json_filenames(self):
        response = self.client.get("/metadata/sample.png")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json().get("detail"),
            "Metadata file must be .json",
        )

    def test_batch_and_grid_require_seed_list(self):
        batch_response = self.client.post(
            "/generate/batch",
            json={"prompt": "x", "seeds": []},
        )
        grid_response = self.client.post(
            "/generate/grid",
            json={"prompt": "x", "seeds": []},
        )

        self.assertEqual(batch_response.status_code, 400)
        self.assertEqual(grid_response.status_code, 400)

    def test_grid_rejects_unsafe_output_paths(self):
        absolute_response = self.client.post(
            "/generate/grid",
            json={"prompt": "x", "seeds": [1], "output": "/tmp/evil.png"},
        )
        traversal_response = self.client.post(
            "/generate/grid",
            json={"prompt": "x", "seeds": [1], "output": "../evil.png"},
        )

        self.assertEqual(absolute_response.status_code, 400)
        self.assertIn("relative", absolute_response.json().get("detail", ""))
        self.assertEqual(traversal_response.status_code, 400)
        self.assertIn("must not contain", traversal_response.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()

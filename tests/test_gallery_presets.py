import json
import tempfile
import unittest
from pathlib import Path

import sana_core.gallery as gallery
import sana_core.presets as presets


class GalleryPresetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.outputs = self.root / "outputs"
        self.presets_dir = self.root / "presets"
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        self.original_gallery_root = gallery.PROJECT_ROOT
        self.original_gallery_output = gallery.OUTPUT_DIR
        self.original_preset_dir = presets.PRESET_DIR
        self.original_preset_path = presets.PRESET_PATH

        gallery.PROJECT_ROOT = self.root
        gallery.OUTPUT_DIR = self.outputs
        presets.PRESET_DIR = self.presets_dir
        presets.PRESET_PATH = self.presets_dir / "prompt_presets.json"

    def tearDown(self):
        gallery.PROJECT_ROOT = self.original_gallery_root
        gallery.OUTPUT_DIR = self.original_gallery_output
        presets.PRESET_DIR = self.original_preset_dir
        presets.PRESET_PATH = self.original_preset_path
        self.temp_dir.cleanup()

    def test_list_gallery_items_reads_metadata_and_skips_invalid_json(self):
        image_path = self.outputs / "sample.png"
        image_path.write_bytes(b"fake-png")

        valid_metadata = {
            "prompt": "test prompt",
            "image_path": "outputs/sample.png",
            "seed": 123,
            "steps": 20,
        }
        (self.outputs / "sample.json").write_text(
            json.dumps(valid_metadata),
            encoding="utf-8",
        )
        (self.outputs / "broken.json").write_text("{broken", encoding="utf-8")

        items = gallery.list_gallery_items()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["prompt"], "test prompt")
        self.assertTrue(items[0]["image_exists"])
        self.assertEqual(items[0]["metadata_path"], "outputs/sample.json")

    def test_save_and_delete_preset_roundtrip(self):
        preset = {
            "id": "custom_test",
            "name": "Custom Test",
            "prompt": "clean product render",
            "negative_prompt": "blurry",
            "width": 512,
            "height": 512,
            "steps": 10,
            "guidance": 4.0,
            "dtype": "float16",
            "tags": ["test"],
        }

        presets.save_preset(preset)
        listed = presets.list_presets()

        self.assertTrue(any(item["id"] == "custom_test" for item in listed))
        self.assertTrue(presets.delete_preset("custom_test"))
        self.assertFalse(presets.delete_preset("missing_preset"))

    def test_gallery_js_uses_dom_text_apis_for_prompt_rendering(self):
        gallery_js = (Path(__file__).resolve().parents[1] / "static" / "gallery.js")
        content = gallery_js.read_text(encoding="utf-8")

        self.assertNotIn("card.innerHTML", content)
        self.assertIn("title.textContent = prompt", content)
        self.assertNotIn("/file/", content)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


class ReleaseGuardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_launch_script_does_not_reference_sana_main_venv(self):
        launch_content = (self.root / "launch.sh").read_text(encoding="utf-8")

        self.assertIn('VENV_PATH=".venv"', launch_content)
        self.assertNotIn("Sana-main/.venv", launch_content)

    def test_smoke_check_scans_expanded_runtime_targets(self):
        smoke_content = (self.root / "smoke_check.sh").read_text(encoding="utf-8")

        for required in (
            "requirements-api.txt",
            "requirements-ui.txt",
            "run_sana_grid.py",
            "api_server.py",
            'grep -R -n "VENV_PATH=.*Sana-main/.venv\\|source .*Sana-main/.venv" launch.sh setup.sh',
        ):
            self.assertIn(required, smoke_content)


if __name__ == "__main__":
    unittest.main()

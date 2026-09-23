import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import scrape_jobs
import triage_agent


class TriageSafetyTests(unittest.TestCase):
    def test_safe_error_detail_redacts_anthropic_key(self):
        detail = triage_agent.safe_error_detail(
            RuntimeError("request rejected for api_key=sk-ant-secret123")
        )
        self.assertNotIn("sk-ant-secret123", detail)
        self.assertIn("[redacted]", detail)

    def test_main_aborts_after_consecutive_model_failures(self):
        jobs = [
            {
                "company": "Example",
                "title": f"Role {index}",
                "location": "Remote",
                "url": f"https://example.com/jobs/{index}",
                "date_posted": f"2026-09-{23 - index:02d}",
                "first_seen": "2026-09-23T00:00:00+00:00",
                "ats": "Test",
            }
            for index in range(8)
        ]
        failing_call = mock.Mock(
            side_effect=RuntimeError("credit balance too low; api_key=sk-ant-secret123")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            scores_path = os.path.join(temp_dir, "scores.json")
            missing_master = os.path.join(temp_dir, "missing-all-jobs.json")
            with (
                mock.patch.object(sys, "argv", ["triage_agent.py", "--limit", "8"]),
                mock.patch.object(triage_agent, "SCORES_PATH", scores_path),
                mock.patch.object(triage_agent, "ALL_JOBS_PATH", missing_master),
                mock.patch.object(triage_agent, "_read_first", side_effect=["profile", "resume"]),
                mock.patch.object(triage_agent, "load_jobs", return_value=jobs),
                mock.patch.object(triage_agent, "load_scores", return_value={"scores": {}}),
                mock.patch.object(triage_agent, "make_call_model", return_value=failing_call),
                mock.patch.object(triage_agent, "fetch_jd", return_value=""),
                mock.patch.object(triage_agent.time, "sleep"),
            ):
                result = triage_agent.main()

            self.assertEqual(1, result)
            self.assertEqual(5, failing_call.call_count)
            with open(scores_path, encoding="utf-8") as handle:
                saved = json.load(handle)["scores"]
            self.assertEqual(5, len(saved))
            first = next(iter(saved.values()))
            self.assertEqual("RuntimeError", first["error_type"])
            self.assertNotIn("sk-ant-secret123", first["error_message"])


class ScraperResilienceTests(unittest.TestCase):
    def test_jobspy_country_aliases(self):
        self.assertEqual("usa", scrape_jobs._jobspy_country("US"))
        self.assertEqual("uk", scrape_jobs._jobspy_country("GB"))
        self.assertEqual(
            "united arab emirates",
            scrape_jobs._jobspy_country("UAE"),
        )

    def test_csu_incomplete_scan_merges_partial_and_previous_jobs(self):
        current = {
            "company": "California State University",
            "title": "Environmental Program Manager",
            "location": "California",
            "url": "https://csucareers.calstate.edu/current",
            "description": "",
        }
        previous = {
            "company": "California State University",
            "title": "Previous Role",
            "location": "California",
            "url": "https://csucareers.calstate.edu/previous",
        }

        def fetch_page(url):
            if "page=1&" in url:
                return '<a class="more-link">more</a>'
            return ""

        with (
            mock.patch.object(scrape_jobs, "CSUCAREERS_MAX_PAGES", 4),
            mock.patch.object(scrape_jobs, "_fetch_csucareers", side_effect=fetch_page),
            mock.patch.object(scrape_jobs, "_parse_csucareers_listing", return_value=[current]),
            mock.patch.object(scrape_jobs, "_load_prev_jobs", return_value=[previous]),
            mock.patch.object(scrape_jobs, "is_mle_role_text", return_value=True),
            mock.patch.object(scrape_jobs.time, "sleep"),
        ):
            jobs = scrape_jobs.scrape_csucareers_recent()

        self.assertEqual(
            {current["url"], previous["url"]},
            {job["url"] for job in jobs},
        )

    def test_unavailable_source_does_not_call_saver(self):
        saver = mock.Mock()

        def unavailable():
            raise scrape_jobs.SourceUnavailableError("blocked")

        result = scrape_jobs._run_guarded_source(unavailable, saver)
        self.assertEqual(2, result)
        saver.assert_not_called()


if __name__ == "__main__":
    unittest.main()

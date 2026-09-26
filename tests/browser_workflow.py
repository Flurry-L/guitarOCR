"""Browser acceptance with deterministic models; run explicitly, without GPUs."""

from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import guitarpro
import pymupdf
from playwright.sync_api import sync_playwright, expect
import uvicorn

from webapp.app import create_app
from webapp.workflow import Workflow

TARGET = "M2 time=4/4 | V0{@0:q:s1f0 @960:q:s2f1 @1920:h:r}"


class BrowserWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

        class DemoWorkflow(Workflow):
            fail_next = False

            def detect(self, sid, mode="auto", source="auto"):
                if self.fail_next:
                    self.fail_next = False
                    raise ValueError("测试用检测失败")
                state = self.load(sid)
                return self.boxes(
                    sid,
                    [
                        {"page": i + 1, "kind": "measure", "bbox": [30, 100, 200, 100], "mode": "tab"}
                        for i in range(len(state["pages"]))
                    ],
                    mode,
                )

        cls.workflow = DemoWorkflow(
            cls.root / "projects", model=cls.root / "missing-model", device="cpu"
        )
        cls.backend_patch = patch("shared.glm_backend.GlmBackend")
        cls.backend_patch.start().return_value.generate.return_value = (TARGET, 20)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.server = uvicorn.Server(
            uvicorn.Config(
                create_app(cls.workflow),
                host="127.0.0.1",
                port=cls.port,
                log_level="error",
            )
        )
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)
        cls.pdf = cls.root / "four-pages.pdf"
        with pymupdf.open() as document:
            for i in range(4):
                page = document.new_page(width=250, height=200)
                page.insert_text((20, 25), f"GuitarOCR example - page {i + 1}")
                for y in range(55, 86, 6):
                    page.draw_line((15, y), (235, y))
                page.insert_text((45, 57), str(i))
            document.save(cls.pdf)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.should_exit = True
        cls.thread.join(10)
        cls.backend_patch.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context(
            viewport={"width": 1440, "height": 1050}, accept_downloads=True
        )
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(f"http://127.0.0.1:{self.port}")
        expect(self.page.locator("#notice")).to_contain_text("识别环境尚未就绪")
        self.page.locator("#files").set_input_files(self.pdf)
        self.page.locator("#upload").click()
        expect(self.page.locator('[data-panel="1"]')).to_have_class(
            "panel active", timeout=15000
        )

    def detect(self):
        self.page.locator("#detect").click()
        expect(self.page.locator("#boxSummary")).to_contain_text(
            "4 个小节", timeout=15000
        )
        expect(self.page.locator("#detect")).to_be_enabled()

    def test_pages_edit_conflict_and_export(self):
        page = self.page
        expect(page.locator("#documentPages")).to_contain_text("4 页")
        page.locator("#nextPage").click()
        expect(page.locator("#pageLabel")).to_contain_text("第 2 / 4 页")
        page.locator("#pageSelect").select_option("3")
        expect(page.locator("#pageLabel")).to_contain_text("第 4 / 4 页")
        self.detect()
        # Move a detected box with actual pointer events.
        page.locator("#pageSelect").select_option("0")
        expect(page.locator("#canvas")).to_have_attribute("aria-busy", "false")
        canvas = page.locator("#canvas").bounding_box()
        width = page.locator("#canvas").evaluate("n => n.width")
        scale = width / 625
        page.mouse.move(canvas["x"] + 90 * scale, canvas["y"] + 140 * scale)
        page.mouse.down()
        page.mouse.move(canvas["x"] + 100 * scale, canvas["y"] + 145 * scale, steps=4)
        page.mouse.up()
        expect(page.locator("#boxSummary")).to_contain_text("未保存")
        page.locator("#saveBoxes").click()
        expect(page.locator("#customTuning")).to_be_hidden()
        page.locator("#tuningPreset").select_option("custom")
        expect(page.locator("#customTuning")).to_be_visible()
        page.locator("#tuning").fill("64,,55")
        page.locator("#saveInfo").click()
        expect(page.locator("#notice")).to_contain_text("1 至 7 个弦")
        page.locator("#tuningPreset").select_option("64,59,55,50,45,40")
        expect(page.locator("#customTuning")).to_be_hidden()
        page.locator("#title").fill("GuitarOCR demo")
        page.locator("#saveInfo").click()
        page.locator("#recognize").click()
        expect(page.locator("#reviewEditor")).to_be_visible(timeout=15000)
        expect(page.locator("#recognize")).to_be_enabled()
        expect(page.locator("#prevMeasure")).to_be_disabled()
        expect(page.locator("#nextIssue")).to_be_hidden()
        page.locator("#eventRows tr").first.locator(".notes").fill("1:7")
        page.locator("#editMode").select_option("text")
        expect(page.locator("#editMode")).to_have_value("table")
        expect(page.locator("#notice")).to_contain_text("请先保存当前修改")
        expect(page.locator("#eventRows tr").first.locator(".notes")).to_have_value("1:7")
        page.locator("#saveMeasure").click()
        expect(page.locator("#notice")).to_contain_text("第 1 小节已保存")
        page.locator("#editMode").select_option("text")
        expect(page.locator("#tableEditor")).to_be_hidden()
        raw = page.locator("#measureText").input_value()
        self.assertTrue(raw.startswith("MEASURE "))
        self.assertNotIn("M2", page.locator("#textEditor").inner_text())
        page.locator("#measureText").fill(raw.replace("s1f7", "s1f6"))
        page.locator("#saveMeasure").click()
        expect(page.locator("#notice")).to_contain_text("第 1 小节已保存")
        expect(page.locator("#eventRows tr").first.locator(".notes")).to_have_value("1:6")
        page.locator("#editMode").select_option("table")
        expect(page.locator("#textEditor")).to_be_hidden()
        # A second tab starts with the same revision, then becomes stale.
        other = self.context.new_page()
        other.goto(page.url)
        expect(other.locator("#reviewEditor")).to_be_visible()
        expect(other.locator("#saveMeasure")).to_be_enabled()
        page.locator("#eventRows tr").first.locator(".notes").fill("1:8")
        page.locator("#saveMeasure").click()
        expect(page.locator("#notice")).to_contain_text("第 1 小节已保存")
        other.locator("#eventRows tr").first.locator(".notes").fill("1:9")
        other.locator("#saveMeasure").click()
        expect(other.locator("#notice")).to_contain_text("其他页面已经修改")
        expect(other.locator("#eventRows tr").first.locator(".notes")).to_have_value(
            "1:9"
        )
        page.locator("#toExport").click()
        page.locator("#export").click()
        expect(page.locator("#downloadGP5")).to_be_visible()
        expect(page.locator("#export")).to_be_hidden()
        with page.expect_download() as text_download:
            page.locator("#downloadScoreText").click()
        self.assertEqual(text_download.value.suggested_filename, "score.txt")
        score_text = self.root / "browser-score.txt"
        text_download.value.save_as(score_text)
        self.assertTrue(score_text.read_text().startswith("MEASURE "))
        self.assertNotIn("M2", score_text.read_text())
        with page.expect_download() as download:
            page.locator("#downloadGP5").click()
        output = self.root / "browser.gp5"
        download.value.save_as(output)
        song = guitarpro.parse(str(output), encoding="cp936")
        self.assertEqual(len(song.tracks[0].measures), 4)
        self.assertEqual(
            song.tracks[0].measures[0].voices[0].beats[0].notes[0].value, 8
        )
        # Newly selected input must not expose the previous project's results.
        page.locator('[data-step="0"]').click()
        page.locator("#files").set_input_files(self.pdf)
        page.locator('[data-step="3"]').click()
        expect(page.locator("#notice")).to_contain_text("先点击「导入乐谱」")
        self.assertFalse(self.errors, self.errors)

    def test_failed_detection_can_be_repaired_and_restored(self):
        self.workflow.fail_next = True
        page = self.page
        page.locator("#detect").click()
        expect(page.locator("#notice")).to_contain_text("测试用检测失败", timeout=15000)
        expect(page.locator("#detect")).to_be_enabled()
        page.locator("#mode").select_option("tab")
        page.locator("#boxTool").select_option("measure")
        expect(page.locator("#canvas")).to_have_attribute("aria-busy", "false")
        canvas = page.locator("#canvas").bounding_box()
        page.mouse.move(canvas["x"] + 30, canvas["y"] + 50)
        page.mouse.down()
        page.mouse.move(canvas["x"] + 180, canvas["y"] + 130, steps=4)
        page.mouse.up()
        page.locator("#saveBoxes").click()
        expect(page.locator('[data-panel="2"]')).to_have_class("panel active")
        page.reload()
        # Startup replaces the restore notice when real models are unavailable.
        expect(page.locator("#notice")).to_contain_text("识别环境尚未就绪")
        expect(page.locator('[data-panel="1"]')).to_have_class("panel active")
        expect(page.locator("#boxSummary")).to_have_text("4 页，1 个小节")
        expect(page.locator("#boxList button")).to_have_count(1)
        expect(page.locator("#canvas")).to_have_attribute("aria-busy", "false")
        self.assertNotIn("测试用检测失败", page.locator("#notice").inner_text())
        self.assertFalse(self.errors, self.errors)

    def test_auto_type_and_per_measure_correction_survive_reload(self):
        page = self.page
        expect(page.locator("#mode")).to_have_value("auto")
        self.detect()
        expect(page.locator("#mode")).to_have_value("auto")
        page.locator("#boxList button").first.click()
        expect(page.locator("#boxMode")).to_have_value("tab")
        page.locator("#boxMode").select_option("notation")
        page.locator("#saveBoxes").click()
        expect(page.locator('[data-panel="2"]')).to_have_class("panel active")
        page.reload()
        expect(page.locator("#mode")).to_have_value("auto")
        page.locator("#boxList button").first.click()
        expect(page.locator("#boxMode")).to_have_value("notation")
        expect(page.locator("#pageLabel")).to_contain_text("五线谱")
        self.assertFalse(self.errors, self.errors)

    def test_review_and_validation_messages(self):
        page = self.page
        self.detect()
        page.locator("#saveBoxes").click()
        page.locator("#title").fill("a" * 501)
        page.locator("#saveInfo").click()
        expect(page.locator("#notice")).to_have_text("曲名最多 500 个字符。")
        page.locator("#title").fill("校对示例")
        page.locator("#saveInfo").click()
        with patch.object(self.workflow.pool, "_generate", return_value=("invalid output", 2)):
            page.locator("#recognize").click()
            expect(page.locator("#reviewEditor")).to_be_visible(timeout=15000)
            expect(page.locator("#recognize")).to_be_enabled()
        expect(page.locator("#fallback")).to_contain_text("暂用休止符占位")
        self.assertNotIn("syntax:", page.locator("#fallback").inner_text())
        expect(page.locator("#nextIssue")).to_be_visible()
        expect(page.locator("#prevMeasure")).to_be_disabled()
        page.locator("#toExport").click()
        expect(page.locator("#export")).to_be_disabled()
        expect(page.locator("#exportWarning")).to_contain_text("4 个小节待检查")
        page.locator('[data-step="3"]').click()
        page.locator("#editMode").select_option("text")
        text = TARGET.replace("M2 ", "MEASURE ", 1)
        page.locator("#measureText").fill(text.replace("s1f0", "s9f0"))
        page.locator("#saveMeasure").click()
        expect(page.locator("#notice")).to_contain_text("弦号超出当前调弦的弦数")
        expect(page.locator("#notice")).to_contain_text("声部 1，第 1 项，第 1 个音符")
        page.locator("#measureText").fill(text)
        page.locator("#saveMeasure").click()
        expect(page.locator("#fallback")).to_be_hidden()
        expect(page.locator("#recognitionSummary")).to_contain_text("3 个待检查")
        self.assertFalse(self.errors, self.errors)


if __name__ == "__main__":
    unittest.main()

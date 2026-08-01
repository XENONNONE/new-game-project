"""Tests for the logging configuration module."""

import json
import logging

from inference_runtime.logging_config import (
    JsonFormatter,
    configure_logging,
    get_logger,
)


class TestJsonFormatter:
    def test_formats_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_includes_exception(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        output = formatter.format(record)
        data = json.loads(output)
        assert data["custom_field"] == "custom_value"


class TestGetLogger:
    def test_returns_avatar_logger(self):
        logger = get_logger()
        assert logger.name == "avatar"

    def test_returns_child_logger(self):
        logger = get_logger("server")
        assert logger.name == "avatar.server"

    def test_logger_is_configured(self):
        logger = get_logger()
        assert len(logger.handlers) > 0


class TestConfigureLogging:
    def test_idempotent(self):
        logger1 = configure_logging()
        handler_count = len(logger1.handlers)
        logger2 = configure_logging()
        assert len(logger2.handlers) == handler_count

    def test_log_level(self):
        logger = configure_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_file_handler(self, tmp_path, monkeypatch):
        log_file = tmp_path / "test.log"
        monkeypatch.setenv("AVATAR_LOG_FILE", str(log_file))
        monkeypatch.setenv("AVATAR_LOG_LEVEL", "INFO")
        # Clear existing handlers to allow reconfiguration
        root = logging.getLogger("avatar")
        root.handlers.clear()
        root = configure_logging("INFO")
        logger = get_logger("test_file")
        logger.info("test message")
        for handler in root.handlers:
            handler.flush()
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Input sanitization and content filtering for safe text generation."""

import re
import unicodedata
from typing import List, Optional, Tuple


class InputSanitizer:
    """Sanitizes and validates user input before passing to the model.

    Strips dangerous characters, enforces length limits, and checks
    against configurable blocked patterns.
    """

    def __init__(
        self,
        max_length: int = 4096,
        blocked_patterns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the sanitizer.

        Args:
            max_length: Maximum allowed input length in characters.
            blocked_patterns: List of regex pattern strings to reject.
        """
        self.max_length = max_length
        self.blocked_patterns: List[re.Pattern[str]] = []
        if blocked_patterns:
            for pattern in blocked_patterns:
                self.blocked_patterns.append(re.compile(pattern, re.IGNORECASE))

    def sanitize(self, text: str) -> str:
        """Sanitize input text by removing dangerous characters.

        Strips null bytes, control characters (except newline and tab),
        and normalizes whitespace runs.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text safe for model consumption.
        """
        # Strip null bytes
        text = text.replace("\x00", "")

        # Remove control characters except newline (\n) and tab (\t)
        cleaned_chars: List[str] = []
        for ch in text:
            if ch in ("\n", "\t"):
                cleaned_chars.append(ch)
            elif unicodedata.category(ch).startswith("C"):
                # Skip control characters (Cc, Cf, Cs, Co, Cn)
                continue
            else:
                cleaned_chars.append(ch)
        text = "".join(cleaned_chars)

        # Normalize runs of whitespace (spaces/tabs) on each line
        # Preserve newlines but collapse consecutive spaces/tabs
        lines = text.split("\n")
        normalized_lines: List[str] = []
        for line in lines:
            normalized_lines.append(re.sub(r"[ \t]+", " ", line).strip())
        text = "\n".join(normalized_lines)

        # Collapse multiple consecutive blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def validate(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate input text against configured rules.

        Checks length limits and blocked patterns.

        Args:
            text: Input text to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None
            if the input is valid.
        """
        if len(text) > self.max_length:
            return False, (
                f"Input exceeds maximum length of {self.max_length} characters "
                f"(got {len(text)})"
            )

        if len(text.strip()) == 0:
            return False, "Input is empty or contains only whitespace"

        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return False, f"Input matches blocked pattern: {pattern.pattern}"

        return True, None


class ContentFilter:
    """Checks input for common prompt injection and adversarial patterns.

    This filter is intentionally conservative — it only blocks obvious
    injection attempts and does not flag legitimate content.
    """

    # Patterns that indicate prompt injection attempts.
    # Each tuple is (compiled_regex, human-readable reason).
    _INJECTION_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"ignore\s+(all\s+)?previous\s+instructions",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to override previous instructions",
        ),
        (
            re.compile(
                r"ignore\s+(all\s+)?above\s+instructions",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to override above instructions",
        ),
        (
            re.compile(
                r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to disregard instructions",
        ),
        (
            re.compile(
                r"(?:^|\n)\s*system\s*prompt\s*:",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to inject a system prompt",
        ),
        (
            re.compile(
                r"you\s+are\s+now\s+(?:in\s+)?(?:a\s+)?(?:new|different|developer|debug|admin)",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to change model persona/mode",
        ),
        (
            re.compile(
                r"enter\s+(?:developer|debug|admin|sudo|god)\s+mode",
                re.IGNORECASE,
            ),
            "Prompt injection: attempt to enter privileged mode",
        ),
    ]

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the content filter.

        Args:
            enabled: Whether the filter is active. When disabled,
                     check_input always returns (True, None).
        """
        self.enabled = enabled

    def check_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check input text for prompt injection patterns.

        Args:
            text: User input to check.

        Returns:
            Tuple of (is_safe, reason). reason is None if input is safe.
        """
        if not self.enabled:
            return True, None

        # Check against known injection patterns
        for pattern, reason in self._INJECTION_PATTERNS:
            if pattern.search(text):
                return False, reason

        # Check for excessive special character density (e.g., encoded payloads)
        # Only flag if the text is long enough and has a very high ratio
        if len(text) > 50:
            special_count = sum(
                1 for ch in text if not (ch.isalnum() or ch.isspace())
            )
            ratio = special_count / len(text)
            if ratio > 0.5:
                return False, (
                    "Suspicious input: excessive special character density "
                    f"({ratio:.0%})"
                )

        return True, None

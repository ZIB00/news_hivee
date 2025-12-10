from typing import Dict, Any, Optional
from .request import call_llm
import os
import logging
import json
import re
import asyncio
from datetime import datetime
from urllib.parse import urlparse
import hashlib
from bs4 import BeautifulSoup
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContentType(Enum):
    RSS = "rss"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"

@dataclass
class ParseResult:
    title: str
    text: str
    url: str
    published_at: str
    source: str
    author: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None
    success: bool = True
    error_reason: Optional[str] = None

class ParserAgent:
    def __init__(self):
        self.cache = {}  # URL cache to avoid duplicate processing
        self.fallback_count = 0  # Track fallback usage
        self.success_count = 0
        self.total_count = 0

        # Spam words to filter out
        self.spam_words = [
            "click here", "buy now", "limited time", "act now",
            "free money", "get rich", "miracle cure", "work from home"
        ]

        # NSFW words to detect
        self.nsfw_words = [
            "porn", "sex", "nude", "adult", "erotic", "explicit",
            "intimate", "sensual", "sexual"
        ]

    def _detect_content_type(self, content: str) -> ContentType:
        """Auto-detect content type (RSS, HTML, plain text)"""
        content_lower = content.lower()

        if '<rss' in content_lower or '<feed' in content_lower:
            return ContentType.RSS
        elif '<html' in content_lower or '<article' in content_lower or '<p>' in content_lower:
            return ContentType.HTML
        else:
            return ContentType.PLAIN_TEXT

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content using BeautifulSoup"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _rule_based_parse(self, content: str, url: str) -> ParseResult:
        """Fallback rule-based parser when LLM fails"""
        content_type = self._detect_content_type(content)

        if content_type == ContentType.HTML:
            clean_text = self._clean_html(content)
        else:
            clean_text = content.strip()

        # Extract title more robustly
        # Remove any leading/trailing whitespace and check for content
        if not clean_text or clean_text.isspace():
            title = "Untitled"
            text = ""
        else:
            # Remove any leading "None" or similar that might be from LLM responses
            clean_text = clean_text.strip()

            # Find the first substantial sentence, skipping explanation text
            sentences = [s.strip() for s in clean_text.split('.') if s.strip()]

            if sentences:
                # Take the first non-trivial sentence as title
                title = sentences[0]
                # If first sentence is too short or looks like explanation, try next
                if len(title) < 10 or title.lower().startswith(('good', 'ok', 'well', 'i', 'here')):
                    title = sentences[0] if len(sentences) > 0 else "Untitled"
                    if len(sentences) > 1:
                        title = sentences[1] if len(sentences[1]) >= len(title) else title
            else:
                title = "Untitled"

            # Ensure title is reasonable
            if not title or len(title.strip()) < 3:
                # Extract first 20-50 chars that look like content
                title = clean_text[:50].strip()
                if len(title) < 3:
                    title = "Untitled"

            # Extract text body (everything after title, or rest of content)
            title_end_pos = clean_text.find(title) if title != "Untitled" else -1
            if title_end_pos != -1:
                text = clean_text[title_end_pos + len(title):].strip()
            else:
                text = clean_text[len(title):].strip()

        # Extract domain as source
        parsed_url = urlparse(url)
        source = parsed_url.netloc or url

        return ParseResult(
            title=title,
            text=text,
            url=url,
            published_at="unknown",
            source=source,
            success=True
        )

    def _validate_result(self, result: ParseResult) -> bool:
        """Validate parsed result quality"""
        # Check title exists and is not empty
        if not result.title or not result.title.strip():
            logger.warning(f"Title is empty: '{result.title}'")
            return False

        # Check title length (minimum 2 characters instead of 3 to be more permissive)
        if len(result.title.strip()) < 2:
            logger.warning(f"Title too short: '{result.title}'")
            return False

        # Check for spam words
        combined_text = f"{result.title} {result.text}".lower()
        for spam_word in self.spam_words:
            if spam_word.lower() in combined_text:
                logger.warning(f"Spam content detected: contains '{spam_word}'")
                return False

        # Check for NSFW content
        for nsfw_word in self.nsfw_words:
            if nsfw_word.lower() in combined_text:
                logger.warning(f"NSFW content detected: contains '{nsfw_word}'")
                return False

        return True

    def _extract_metadata(self, content: str, url: str) -> Dict[str, Any]:
        """Extract metadata like author, language, country"""
        metadata = {}

        # Extract domain as source
        parsed_url = urlparse(url)
        metadata['source'] = parsed_url.netloc

        # Try to extract author from content
        author_pattern = r'(?:by|author)[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        author_match = re.search(author_pattern, content, re.IGNORECASE)
        if author_match:
            metadata['author'] = author_match.group(1)

        # Determine language based on common patterns (could be improved with langdetect)
        # This is a simplified version - could use langdetect library for accuracy
        russian_chars = sum(1 for c in content if ord(c) > 1000)  # Cyrillic range
        total_chars = len([c for c in content if c.isalpha()])

        if total_chars > 0 and russian_chars / total_chars > 0.3:
            metadata['language'] = 'ru'
        else:
            metadata['language'] = 'en'

        return metadata

    async def _call_llm_parser(self, text: str, url: str) -> Optional[Dict[str, Any]]:
        """Call LLM for parsing with proper error handling"""
        prompt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'prompts', 'parser_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        safe_text = text.replace("{", "{{").replace("}", "}}")
        prompt = prompt_template.format(text=safe_text, url=url)

        logger.info(f"Parser: Sending text to LLM: length {len(text)} chars, start: {text[:500]}...")
        response = await call_llm(prompt)

        if response == "error":
            return None

        try:
            logger.info(f"Parser: Received response from LLM: {response[:200]}...")

            # Find JSON in response
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Find first valid JSON object in response
                start_indices = [i for i, c in enumerate(response) if c == '{']
                found_jsons = []

                for start in start_indices:
                    brace_count = 0
                    for i in range(start, len(response)):
                        if response[i] == '{':
                            brace_count += 1
                        elif response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                potential_json = response[start:i+1]
                                try:
                                    json.loads(potential_json)
                                    found_jsons.append(potential_json)
                                except json.JSONDecodeError:
                                    pass
                                break

                if found_jsons:
                    json_str = found_jsons[-1]
                else:
                    return None

            parsed_data = json.loads(json_str)
            return parsed_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing JSON from parser: {response}, error: {e}")
            return None

    async def parse_with_retry(self, text: str, url: str, max_retries: int = 3) -> ParseResult:
        """Parse with retry mechanism and fallback strategies"""
        self.total_count += 1

        # Check cache first
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.cache:
            cached_result = self.cache[url_hash]
            logger.info(f"Cache hit for URL: {url}")
            return cached_result

        # Validate input
        if not text or len(text.strip()) == 0:
            return ParseResult(
                title="Empty Content",
                text="",
                url=url,
                published_at="unknown",
                source="unknown",
                success=False,
                error_reason="Empty content provided"
            )

        # Attempt parsing with LLM
        attempt = 0
        while attempt < max_retries:
            llm_result = await self._call_llm_parser(text, url)

            if llm_result:
                # Convert to ParseResult
                result = ParseResult(
                    title=llm_result.get("title", ""),
                    text=llm_result.get("text", ""),
                    url=url,
                    published_at=llm_result.get("published_at", "unknown"),
                    source=llm_result.get("source", "unknown"),
                    author=llm_result.get("author"),
                    language=llm_result.get("language"),
                    country=llm_result.get("country")
                )

                # Add extracted metadata if not provided by LLM
                if not result.author or not result.source:
                    metadata = self._extract_metadata(text, url)
                    result.author = result.author or metadata.get('author')
                    result.language = result.language or metadata.get('language')
                    result.source = result.source or metadata.get('source')

                # Validate the result
                if self._validate_result(result):
                    self.success_count += 1
                    self.cache[url_hash] = result
                    return result
                else:
                    logger.warning(f"LLM result failed validation on attempt {attempt + 1}")
            else:
                logger.warning(f"LLM returned error on attempt {attempt + 1}")

            attempt += 1
            if attempt < max_retries:
                await asyncio.sleep(0.5 * attempt)  # Exponential backoff

        # All LLM attempts failed, use rule-based fallback
        logger.warning(f"All LLM attempts failed for {url}, using rule-based fallback")
        self.fallback_count += 1
        result = self._rule_based_parse(text, url)
        result.success = True  # Rule-based parsing is considered successful
        self.success_count += 1
        self.cache[url_hash] = result

        return result

    def get_metrics(self) -> Dict[str, float]:
        """Return quality metrics"""
        if self.total_count == 0:
            return {"fallback_rate": 0.0, "success_rate": 0.0}

        fallback_rate = self.fallback_count / self.total_count
        success_rate = self.success_count / self.total_count

        return {
            "fallback_rate": fallback_rate,
            "success_rate": success_rate,
            "total_processed": self.total_count
        }

# Global instance to maintain state
parser_agent_instance = ParserAgent()

async def run(text: str, url: str) -> Dict[str, Any]:
    """Main run function maintaining backward compatibility"""
    result = await parser_agent_instance.parse_with_retry(text, url)

    # Log metrics periodically
    if parser_agent_instance.total_count % 10 == 0:
        metrics = parser_agent_instance.get_metrics()
        logger.info(f"Parser metrics: {metrics}")

    return {
        "title": result.title,
        "text": result.text,
        "url": result.url,
        "published_at": result.published_at,
        "source": result.source,
        "author": result.author,
        "language": result.language
    }
"""Utilities for loading Markdown, PDF and URL content into the knowledge base."""

from __future__ import annotations

import io
import os
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def load_markdown(file_stream: io.BytesIO) -> str:
    content = file_stream.read().decode("utf-8", errors="ignore")
    return content.strip()


def _looks_like_toc_line(line: str) -> bool:
    if not line:
        return False
    stripped = line.strip()
    if len(stripped) < 6:
        return False
    if re.search(r"([.．·•]{4,}|…{2,})", stripped):
        return True
    if re.search(r"^[^\n]{1,80}\s\d{1,3}$", stripped):
        return True
    return False


def _looks_like_catalog_page(text: str) -> bool:
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    toc_like = sum(1 for line in lines if _looks_like_toc_line(line))
    page_marker_like = sum(1 for line in lines if re.match(r"^(=+\s*)?第?\s*\d+\s*页", line))
    return toc_like >= 4 or (toc_like >= max(2, len(lines) // 2)) or page_marker_like >= max(3, len(lines) // 2)


def _is_low_information_text(text: str) -> bool:
    if not text:
        return True

    compact = re.sub(r"\s+", "", text)
    if len(compact) < 20:
        return True

    meaningful = re.findall(r"[\u4e00-\u9fffA-Za-z]", compact)
    meaningful_ratio = len(meaningful) / max(len(compact), 1)
    dotted_ratio = len(re.findall(r"[.．·•]", compact)) / max(len(compact), 1)
    return meaningful_ratio < 0.25 or dotted_ratio > 0.20


def clean_pdf_text(text: str) -> str:
    text = text.replace("\ufffd", "")
    text = re.sub(r"[.．·•]{4,}", " ", text)
    text = re.sub(r"[_-]{5,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<![\s])\.(?![\s])", ". ", text)
    text = re.sub(r"(?<![\s])\,(?![\s])", ", ", text)
    return text.strip()


def _normalize_pdf_page_text(text: str) -> str:
    if not text:
        return ""

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(第?\s*\d+\s*页|\d+/\d+|\d+)$", line):
            continue
        if _looks_like_toc_line(line):
            continue
        lines.append(line)

    merged = "\n".join(lines).strip()
    merged = clean_pdf_text(merged)
    if _looks_like_catalog_page(merged) or _is_low_information_text(merged):
        return ""
    return merged


def has_meaningful_content(text: str) -> bool:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))
    visible_chars = len(re.sub(r"\s+", "", text))
    language_chars = chinese_chars + english_chars
    return language_chars > 20 and visible_chars > 40 and (language_chars / max(visible_chars, 1)) > 0.25


def load_pdf(file_input) -> str:
    """Extract PDF text with extra filtering for catalog pages and low-quality noise."""
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            print(f"PDF file not found: {file_input}")
            return ""
        try:
            with open(file_input, "rb") as f:
                file_stream = io.BytesIO(f.read())
        except Exception as e:
            print(f"Failed to open PDF file: {str(e)}")
            return ""
    else:
        file_stream = file_input
        file_stream.seek(0)

    if PdfReader is not None:
        try:
            reader = PdfReader(file_stream)
            texts = []
            all_text = ""
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                    all_text += text + "\n"
                    text = _normalize_pdf_page_text(text)
                    if text:
                        texts.append(f"=== 第{page_num}页 ===\n{text}")
                except Exception as e:
                    print(f"PyPDF2 page parse failed: {str(e)}")

            extracted_text = "\n\n".join(texts).strip()
            if extracted_text:
                print(f"PyPDF2提取文本长度: {len(extracted_text)}字符")
                if len(extracted_text) > 100 and has_meaningful_content(extracted_text):
                    print("PyPDF2提取文本质量良好")
                    return extracted_text
                print("PyPDF2提取文本质量一般，但仍返回")
                return extracted_text
            print(f"PyPDF2原始文本长度: {len(all_text)}字符")
        except Exception as e:
            print(f"PyPDF2解析失败: {str(e)}")

    file_stream.seek(0)
    if pdfplumber is not None:
        try:
            texts = []
            all_text = ""
            with pdfplumber.open(file_stream) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                        all_text += text + "\n"
                        text = _normalize_pdf_page_text(text)
                        if text:
                            texts.append(f"=== 第{page_num}页 ===\n{text}")
                    except Exception as e:
                        print(f"pdfplumber page parse failed: {str(e)}")

            extracted_text = "\n\n".join(texts).strip()
            if extracted_text:
                print(f"pdfplumber提取文本长度: {len(extracted_text)}字符")
                if len(extracted_text) > 100 and has_meaningful_content(extracted_text):
                    print("pdfplumber提取文本质量良好")
                    return extracted_text
                print("pdfplumber提取文本质量一般，但仍返回")
                return extracted_text
            print(f"pdfplumber原始文本长度: {len(all_text)}字符")
        except Exception as e:
            print(f"pdfplumber解析失败: {str(e)}")

    file_stream.seek(0)
    if PdfReader is not None:
        try:
            reader = PdfReader(file_stream)
            texts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = _normalize_pdf_page_text(page.extract_text() or "")
                    if text:
                        texts.append(f"=== 第{page_num}页 ===\n{text}")
                except Exception as e:
                    print(f"Raw PyPDF2 page parse failed: {str(e)}")
            extracted_text = "\n\n".join(texts).strip()
            print(f"Raw PyPDF2提取文本长度: {len(extracted_text)}字符")
            return extracted_text
        except Exception as e:
            print(f"Raw PyPDF2解析失败: {str(e)}")

    print("Warning: all PDF parsing methods failed")
    return ""


def normalize_url(url: str) -> str:
    return url.strip()


def fetch_url(url: str, timeout: int = 15):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text
    title = ""
    if trafilatura:
        extracted = trafilatura.extract(html, url=url, with_metadata=True)
        if extracted:
            text = extracted.strip()
            try:
                meta = trafilatura.metadata.extract_metadata(html, url=url)  # type: ignore
                if meta and getattr(meta, "title", None):
                    title = meta.title
            except Exception:
                pass
            return text, {"url": url, "title": title}

    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    return text.strip(), {"url": url, "title": title}

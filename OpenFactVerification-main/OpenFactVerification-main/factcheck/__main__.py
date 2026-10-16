"""Command-line fact-checking.

Examples (run from the project root with the virtual environment active):
  python -m factcheck --modal string --input "Kampala is the capital of Uganda." --country Uganda
  python -m factcheck --modal text --input demo_data/text.txt
  python -m factcheck --modal url --input https://example.com/article
  python -m factcheck --modal image --input demo_data/image.webp
  python -m factcheck --modal audio --input demo_data/speech.mp3   (local faster-whisper transcription)
"""

import json
import argparse
import os
import sys

from factcheck.utils.llmclient import CLIENTS, LLMError
from factcheck.utils.utils import load_yaml
from factcheck.utils.transcription import get_transcriber
from factcheck.core.Retriever import RetrievalError
from factcheck import FactCheck, inputs


def check(args):
    api_config = {}
    if args.api_config:
        try:
            api_config = load_yaml(args.api_config) or {}
        except Exception as e:
            print(f"Error loading api config: {e}")

    factcheck = FactCheck(
        default_model=args.model, client=args.client, api_config=api_config, prompt=args.prompt, retriever=args.retriever
    )

    transcriber = get_transcriber() if args.modal in ("speech", "audio", "video") else None
    modal = {"speech": "audio", "string": "text_string"}.get(args.modal, args.modal)
    if modal == "text_string":
        media = inputs.prepare_text(args.input)
    elif modal == "text":
        with open(args.input, encoding="utf-8") as f:
            media = inputs.prepare_text(f.read())
    elif modal == "url":
        media = inputs.prepare_url(args.input)
    else:
        media = inputs.prepare_media(modal, args.input, vision_client=factcheck.vision_client, transcriber=transcriber,
                                     language=args.language, progress=lambda m: print(m, file=sys.stderr))

    res = factcheck.check_text(media.text, country=args.country, language=args.language, output_language=args.output_language)
    res["input"] = {"type": modal, "extracted": media.extracted, "notes": media.notes}
    print(json.dumps(res, indent=2, ensure_ascii=False))


def main():
    # Windows consoles default to a legacy code page; force UTF-8 so Swahili/Luganda text prints safely.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Kweli Lens AI command-line fact-checker")
    parser.add_argument("--model", type=str, default=os.environ.get("TRUTHLENS_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--client", type=str, default=None, choices=CLIENTS.keys())
    parser.add_argument("--prompt", type=str, default="truthlens_prompt")
    parser.add_argument("--retriever", type=str, default="serper")
    parser.add_argument("--modal", type=str, default="text", choices=["string", "text", "url", "image", "speech", "audio", "video"])
    parser.add_argument("--input", type=str, default="demo_data/text.txt")
    parser.add_argument("--country", type=str, default=None)
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "en", "sw", "lg"])
    parser.add_argument("--output_language", type=str, default="en", choices=["en", "sw", "lg"])
    parser.add_argument("--api_config", type=str, default=None, help="Optional YAML file with API keys")
    args = parser.parse_args()
    try:
        check(args)
    except (inputs.InputError, RetrievalError, LLMError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Mentor chat — упрощённая схема (без баз и индексов).

1. Ищет ключевые слова вопроса в файлах data/knowledge/*.txt|*.md
2. Берёт лучший абзац и передаёт его в обученную модель
   (через локальный POST /api/chat) со строгой инструкцией
3. Обрезает ответ до 3 предложений и следит за строкой «Мораль: ...»

Запуск:
    py -3 scripts/mentor_chat.py "твой вопрос"
    py -3 scripts/mentor_chat.py            # режим REPL
"""

import glob
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOW_DIR = os.path.join(ROOT, "data", "knowledge")
READ_DIR = os.path.join(ROOT, "read")  # папка пользователя (любые файлы)
CHAT_URL = "http://127.0.0.1:8000/api/chat"

INSTRUCTION = ("Используя этот текст, ответь пользователю кратко "
               "(максимум 3 предложения), в стиле прямого наставника "
               "с легкой иронией, и обязательно добавь в конце строку "
               "'Мораль: ...'")

RE_SENT = re.compile(r"(?<=[.!?…])\s+")
RE_WORDS = re.compile(r"[а-яёa-z]{4,}", re.I)


def load_paragraphs() -> list[tuple[str, str]]:
    """Абзацы из data/knowledge/*.txt|*.md и из read/ (любые файлы)."""
    out = []
    paths = sorted(glob.glob(os.path.join(KNOW_DIR, "*.txt")) +
                   glob.glob(os.path.join(KNOW_DIR, "*.md")) +
                   [p for p in glob.glob(os.path.join(READ_DIR, "*"))
                    if os.path.isfile(p)])
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not text.strip():
            print(f"  пропуск пустого файла: {os.path.basename(path)}")
            continue
        name = os.path.basename(path)
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) >= 40:
                out.append((name, para))
    return out, paths


def _stems(words: list[str]) -> set[str]:
    """Простые «корни»: первые 6 букв — чтобы падежи совпадали
    (искусстве/искусство -> искусс)."""
    return {w[:6] if len(w) >= 6 else w for w in words}


def search(query: str, paras: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Обычный поиск: сколько корней слов вопроса встречается в абзаце."""
    q_stems = _stems(RE_WORDS.findall(query.lower()))
    if not q_stems:
        return None
    best, best_score = None, 0
    for src, para in paras:
        score = len(q_stems & _stems(RE_WORDS.findall(para.lower())))
        if score > best_score:
            best, best_score = (src, para), score
    return best if best_score > 0 else None


def ask_model(prompt: str) -> str:
    body = json.dumps({"message": prompt}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("reply", "").strip()
    except Exception as e:  # noqa: BLE001
        return f"[модель недоступна: {e}] — запустите: npm run ai"


def polish(reply: str, source_text: str) -> str:
    """Максимум 3 предложения; если модель не добавила «Мораль» — добавляем
    заглушку из исходного текста (пока модель не следует инструкциям дословно)."""
    sents = [s for s in RE_SENT.split(reply.replace("\n", " ")) if s.strip()]
    reply = " ".join(sents[:3]).strip()
    if "Мораль" not in reply:
        moral_src = RE_SENT.split(source_text.strip())[-1].strip()
        reply += f"\nМораль: {moral_src}"
    return reply


def answer(query: str, paras: list[tuple[str, str]]) -> str:
    hit = search(query, paras)
    if hit is None:
        return ("Мораль: В шпаргалках не нашлось ничего по этой теме — "
                "уточни вопрос или добавь текст в data/knowledge/.")
    src, text = hit
    prompt = (f"{INSTRUCTION}\n\n"
              f"Текст ({src}):\n{text}\n\n"
              f"Вопрос: {query}\nОтвет:")
    reply = polish(ask_model(prompt), text)
    return f"{reply}\n\n[i источник: {src}, score по ключевым словам]"


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    paras, paths = load_paragraphs()
    if not paths:
        raise SystemExit(f"Нет файлов-шпаргалок в {KNOW_DIR} — положи туда .txt")
    print(f"шпаргалки: {len(paths)} файл(ов), {len(paras)} абзацев")
    if query:
        print(answer(query, paras))
        return
    print("MENTOR-чат (REPL). 'exit' — выход.")
    while True:
        try:
            q = input("\nВы> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit", "выход"):
            break
        print("\nНаставник> " + answer(q, paras))


if __name__ == "__main__":
    main()
